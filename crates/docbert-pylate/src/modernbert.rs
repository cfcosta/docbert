use core::f32;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Embedding,
    LayerNorm,
    Linear,
    Module,
    VarBuilder,
    embedding,
    layer_norm_no_bias,
    linear,
    linear_no_bias,
    ops::{softmax, softmax_last_dim},
};
use serde::Deserialize;

// This module has been adapted from the `candle` library in order to properly fit the PyLate format.

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    #[serde(default)]
    #[serde(flatten)]
    pub classifier_config: Option<ClassifierConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Copy, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierPooling {
    #[default]
    CLS,
    MEAN,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ClassifierConfig {
    pub id2label: HashMap<String, String>,
    pub label2id: HashMap<String, String>,
    pub classifier_pooling: ClassifierPooling,
}

/// Single-slot last-used cache for per-batch GPU tensors.
///
/// The tensors cached during a forward pass (packed rope tables, packed
/// position ids, local attention masks) are reused heavily *within* one
/// pass — up to all 22 layers share the same key — but a long indexing
/// run sees a near-unique key per batch, so a grow-only map slowly
/// leaks VRAM (the reason `release_cached_device_memory` exists). One
/// slot keeps the intra-forward reuse and caps the footprint at a
/// single entry.
#[derive(Debug)]
struct LastUsedCache<K, V>(Mutex<Option<(K, V)>>);

impl<K, V: Clone> LastUsedCache<K, V> {
    fn new() -> Self {
        Self(Mutex::new(None))
    }

    /// Returns the value cached under `key`, or builds, caches, and
    /// returns it, evicting whatever was cached before.
    fn get_or_try_insert<Q>(
        &self,
        key: &Q,
        build: impl FnOnce() -> Result<V>,
    ) -> Result<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: PartialEq + ToOwned<Owned = K> + ?Sized,
    {
        let mut slot = self.0.lock().unwrap();
        if let Some((cached_key, value)) = slot.as_ref() {
            if cached_key.borrow() == key {
                return Ok(value.clone());
            }
        }
        let value = build()?;
        *slot = Some((key.to_owned(), value.clone()));
        Ok(value)
    }
}

/// Cache of the last-used packed cos/sin tables keyed by the list of
/// valid sequence lengths.
#[cfg(feature = "cuda")]
type PackedCosSinCache = Arc<LastUsedCache<Vec<usize>, (Tensor, Tensor)>>;

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    #[cfg(feature = "cuda")]
    packed_cos_sin: PackedCosSinCache,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        config: &Config,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let max_seq_len = config.max_position_embeddings;
        // Angles are always computed in F32: a half-precision position
        // index quantizes badly (position 300 in BF16 has a ULP of 2),
        // while the resulting sin/cos values live in [-1, 1] and cast to
        // the trunk dtype without meaningful loss.
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            #[cfg(feature = "cuda")]
            packed_cos_sin: Arc::new(LastUsedCache::new()),
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(
            &q.contiguous()?,
            &self.cos,
            &self.sin,
        )?;
        let k_embed = candle_nn::rotary_emb::rope(
            &k.contiguous()?,
            &self.cos,
            &self.sin,
        )?;
        Ok((q_embed, k_embed))
    }

    #[cfg(feature = "cuda")]
    fn apply_rotary_emb_thd(
        &self,
        q: &Tensor,
        k: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q = if q.is_contiguous() {
            q.clone()
        } else {
            q.contiguous()?
        };
        let k = if k.is_contiguous() {
            k.clone()
        } else {
            k.contiguous()?
        };
        let q_embed =
            candle_nn::rotary_emb::rope_thd(&q, &self.cos, &self.sin)?;
        let k_embed =
            candle_nn::rotary_emb::rope_thd(&k, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }

    /// Applies rope to `(total_tokens, heads, head_dim)` packed q/k.
    ///
    /// `rope_thd` applies cos/sin row `i` to token row `i`, so the
    /// fused kernel works on a packed layout as long as the tables are
    /// gathered into the same packed order first — that gather is what
    /// `positions` encodes, and the gathered tables are cached because
    /// every layer of a forward pass shares them.
    #[cfg(feature = "cuda")]
    fn apply_rotary_emb_packed(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
        valid_lens: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin) =
            self.packed_cos_sin.get_or_try_insert(valid_lens, || {
                let cos = self.cos.index_select(positions, 0)?;
                let sin = self.sin.index_select(positions, 0)?;
                Ok((cos, sin))
            })?;
        let q = q.unsqueeze(0)?;
        let k = k.unsqueeze(0)?;
        let q = if q.is_contiguous() {
            q
        } else {
            q.contiguous()?
        };
        let k = if k.is_contiguous() {
            k
        } else {
            k.contiguous()?
        };
        let q_embed =
            candle_nn::rotary_emb::rope_thd(&q, &cos, &sin)?.squeeze(0)?;
        let k_embed =
            candle_nn::rotary_emb::rope_thd(&k, &cos, &sin)?.squeeze(0)?;
        Ok((q_embed, k_embed))
    }
}

/// FlashAttention kernels only ship for F16/BF16, so a half-precision
/// trunk passes through unchanged while an F32 trunk round-trips via
/// F16. The F16 round-trip is safe even though a full F16 *trunk*
/// overflows: attention inputs are post-LayerNorm and its outputs are
/// convex combinations of V rows, so values stay far from ±65504.
#[cfg(feature = "cuda")]
fn flash_compat_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => dtype,
        _ => DType::F16,
    }
}

#[derive(Clone)]
struct ModernBertAttention {
    qkv: Linear,
    proj: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl ModernBertAttention {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size =
            config.hidden_size / config.num_attention_heads;

        let qkv = {
            let qkv = linear_no_bias(
                config.hidden_size,
                config.hidden_size * 3,
                vb.pp("Wqkv"),
            )?;
            let q_scale = (attention_head_size as f64).powf(-0.5);
            let q_weight =
                (qkv.weight().narrow(0, 0, config.hidden_size)? * q_scale)?;
            let k_weight = qkv.weight().narrow(
                0,
                config.hidden_size,
                config.hidden_size,
            )?;
            let v_weight = qkv.weight().narrow(
                0,
                config.hidden_size * 2,
                config.hidden_size,
            )?;
            let weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
            Linear::new(weight, None)
        };
        let proj = linear_no_bias(
            config.hidden_size,
            config.hidden_size,
            vb.pp("Wo"),
        )?;

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((
                b,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?;

        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        let att = att.broadcast_add(attention_mask)?;
        let att = softmax_last_dim(&att)?;

        let xs = att.matmul(&v)?;

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }

    #[cfg(feature = "cuda")]
    fn forward_unmasked(
        &self,
        hidden_states: &Tensor,
        local_window: Option<usize>,
    ) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        let qkv = xs.apply(&self.qkv)?;

        let q = qkv.narrow(2, 0, d)?.reshape((
            b,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let k = qkv.narrow(2, d, d)?.reshape((
            b,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let v = qkv.narrow(2, d * 2, d)?.reshape((
            b,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_thd(&q, &k)?;
        let orig_dtype = q.dtype();
        let flash_dtype = flash_compat_dtype(orig_dtype);
        let q = q.to_dtype(flash_dtype)?;
        let k = k.to_dtype(flash_dtype)?;
        let v = v.to_dtype(flash_dtype)?;
        let xs = match local_window {
            Some(window) => candle_flash_attn::flash_attn_windowed(
                &q,
                &k,
                &v,
                1.0,
                Some(window),
                Some(window),
            )?,
            None => candle_flash_attn::flash_attn(&q, &k, &v, 1.0, false)?,
        };
        let xs = xs.to_dtype(orig_dtype)?;

        let xs = xs.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }

    /// Varlen flash attention on `(total_tokens, hidden)` packed input.
    ///
    /// Both global and local (sliding-window) layers run directly on
    /// the packed layout: rope goes through the gathered-table
    /// `rope_thd` fast path and the window, when present, is handled
    /// inside the flash kernel. Nothing is unpacked or repacked per
    /// layer.
    #[cfg(feature = "cuda")]
    fn forward_varlen_fully_packed(
        &self,
        packed_hidden_states: &Tensor,
        positions: &Tensor,
        valid_lens: &[usize],
        seqlens: &Tensor,
        max_seq_len: usize,
        local_window: Option<usize>,
    ) -> Result<Tensor> {
        let (total_tokens, d) = packed_hidden_states.dims2()?;
        let qkv = packed_hidden_states.apply(&self.qkv)?;

        let q = qkv.narrow(1, 0, d)?.reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let k = qkv.narrow(1, d, d)?.reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let v = qkv.narrow(1, d * 2, d)?.reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;

        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_packed(&q, &k, positions, valid_lens)?;
        let orig_dtype = q.dtype();
        let flash_dtype = flash_compat_dtype(orig_dtype);
        let q = q.to_dtype(flash_dtype)?;
        let k = k.to_dtype(flash_dtype)?;
        let v = v.to_dtype(flash_dtype)?;
        let xs = match local_window {
            Some(window) => candle_flash_attn::flash_attn_varlen_windowed(
                &q,
                &k,
                &v,
                seqlens,
                seqlens,
                max_seq_len,
                max_seq_len,
                1.0,
                Some(window),
                Some(window),
            )?,
            None => candle_flash_attn::flash_attn_varlen(
                &q,
                &k,
                &v,
                seqlens,
                seqlens,
                max_seq_len,
                max_seq_len,
                1.0,
                false,
            )?,
        };
        let xs = xs.to_dtype(orig_dtype)?;

        let xs = xs.reshape((total_tokens, d))?;
        xs.apply(&self.proj)
    }

    /// Flash attention for fixed-length query batches: every padded
    /// row stays a query (ColBERT expansion rows' outputs feed
    /// MaxSim), but only each sequence's valid prefix serves as
    /// keys/values — mirroring the eager path's additive mask, which
    /// only masks key columns.
    #[cfg(feature = "cuda")]
    fn forward_query_varlen(
        &self,
        hidden_states: &Tensor,
        valid_lens: &[usize],
        seqlens_q: &Tensor,
        seqlens_k: &Tensor,
        max_seqlen_k: usize,
        local_window: Option<usize>,
    ) -> Result<Tensor> {
        let (b, seq_len, d) = hidden_states.dims3()?;
        let qkv = hidden_states.apply(&self.qkv)?;

        let q = qkv.narrow(2, 0, d)?.reshape((
            b,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let k = qkv.narrow(2, d, d)?.reshape((
            b,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let v = qkv.narrow(2, d * 2, d)?.reshape((
            b,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_thd(&q, &k)?;
        let orig_dtype = q.dtype();
        let flash_dtype = flash_compat_dtype(orig_dtype);
        let q = q
            .reshape((
                b * seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .to_dtype(flash_dtype)?;
        let k = pack_varlen_thd(&k, valid_lens)?.to_dtype(flash_dtype)?;
        let v = pack_varlen_thd(&v, valid_lens)?.to_dtype(flash_dtype)?;
        let xs = match local_window {
            Some(window) => candle_flash_attn::flash_attn_varlen_windowed(
                &q,
                &k,
                &v,
                seqlens_q,
                seqlens_k,
                seq_len,
                max_seqlen_k,
                1.0,
                Some(window),
                Some(window),
            )?,
            None => candle_flash_attn::flash_attn_varlen(
                &q,
                &k,
                &v,
                seqlens_q,
                seqlens_k,
                seq_len,
                max_seqlen_k,
                1.0,
                false,
            )?,
        };
        let xs = xs.to_dtype(orig_dtype)?;

        let xs = xs.reshape((b, seq_len, d))?;
        xs.apply(&self.proj)
    }
}

#[derive(Clone)]
pub struct ModernBertMLP {
    wi: Linear,
    wo: Linear,
}

impl ModernBertMLP {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        let wo = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("Wo"),
        )?;
        Ok(Self { wi, wo })
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let xs = xs.chunk(2, D::Minus1)?;
        let xs = (&xs[0].gelu_erf()? * &xs[1])?.apply(&self.wo)?; // GeGLU
        Ok(xs)
    }
}

/// Loads one of ModernBERT's bias-free LayerNorms, pairing it with a
/// persistent zero bias on CUDA.
///
/// `candle_nn::LayerNorm` only dispatches its fused CUDA kernel when a
/// bias tensor is present; without one every norm takes a composed
/// multi-kernel path that round-trips half-precision inputs through
/// F32. `x * w + 0` is exact and the fused kernel accumulates in F32
/// internally, so the zero bias changes nothing numerically. On CPU
/// there is no fused path to unlock and the extra bias add is pure
/// waste, so the plain no-bias constructor stays.
fn layer_norm_no_bias_fused(
    size: usize,
    eps: f64,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    if !vb.device().is_cuda() {
        return layer_norm_no_bias(size, eps, vb);
    }
    let weight = vb.get(size, "weight")?;
    let bias = Tensor::zeros(size, weight.dtype(), weight.device())?;
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Clone)]
pub struct ModernBertLayer {
    attn: ModernBertAttention,
    mlp: ModernBertMLP,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl ModernBertLayer {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
    ) -> Result<Self> {
        let attn =
            ModernBertAttention::load(vb.pp("attn"), config, rotary_emb)?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;
        let attn_norm = layer_norm_no_bias_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attn_norm"),
        )
        .ok();
        let mlp_norm = layer_norm_no_bias_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("mlp_norm"),
        )?;
        Ok(Self {
            attn,
            mlp,
            attn_norm,
            mlp_norm,
            uses_local_attention,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let xs = self.attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }

    #[cfg(feature = "cuda")]
    fn forward_unmasked(
        &self,
        xs: &Tensor,
        local_window: Option<usize>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let xs = self.attn.forward_unmasked(&xs, local_window)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn forward_varlen_packed_input(
        &self,
        packed_xs: &Tensor,
        valid_lens: &[usize],
        positions: &Tensor,
        seqlens: &Tensor,
        max_seq_len: usize,
        local_window: Option<usize>,
    ) -> Result<Tensor> {
        let mut xs = packed_xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }
        let attn_out = self.attn.forward_varlen_fully_packed(
            &xs,
            positions,
            valid_lens,
            seqlens,
            max_seq_len,
            local_window,
        )?;
        let xs = (attn_out + packed_xs)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        xs + mlp_out
    }

    #[cfg(feature = "cuda")]
    fn forward_query_varlen(
        &self,
        xs: &Tensor,
        valid_lens: &[usize],
        seqlens_q: &Tensor,
        seqlens_k: &Tensor,
        max_seqlen_k: usize,
        local_window: Option<usize>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let xs = self.attn.forward_query_varlen(
            &xs,
            valid_lens,
            seqlens_q,
            seqlens_k,
            max_seqlen_k,
            local_window,
        )?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertHead {
    dense: Linear,
    norm: LayerNorm,
}

impl ModernBertHead {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear_no_bias(
            config.hidden_size,
            config.hidden_size,
            vb.pp("dense"),
        )?;
        let norm = layer_norm_no_bias_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("norm"),
        )?;
        Ok(Self { dense, norm })
    }
}

impl Module for ModernBertHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.dense)?.gelu_erf()?.apply(&self.norm)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertDecoder {
    decoder: Linear,
}

impl ModernBertDecoder {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // The decoder weights are tied with the embeddings layer weights
        let decoder_weights = vb.get(
            (config.vocab_size, config.hidden_size),
            "embeddings.tok_embeddings.weight",
        )?;
        let decoder_bias = vb.get(config.vocab_size, "decoder.bias")?;
        let decoder = Linear::new(decoder_weights, Some(decoder_bias));
        Ok(Self { decoder })
    }
}

impl Module for ModernBertDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.decoder)?;
        Ok(xs)
    }
}

// Global attention mask calculated from padded token inputs
fn prepare_4d_attention_mask(
    mask: &Tensor,
    dtype: DType,
    tgt_len: Option<usize>,
) -> Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .to_dtype(dtype)?
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?;

    let inverted_mask = (1.0 - expanded_mask)?;

    // The additive mask must stay FINITE in the target dtype. Padding
    // rows in sliding-window layers can have every in-window column
    // masked, and an all–minus-infinity row softmaxes to NaN, which then
    // contaminates the whole batch through K/V in the next layer. With a
    // large-but-finite value those rows degrade to uniform attention over
    // masked columns instead — harmless, their outputs are discarded —
    // while exp(min - rowmax) still underflows to exactly 0 in valid
    // rows. f32::MIN overflows to -inf when cast to F16/BF16, so those
    // dtypes need their own constants.
    let min_value = match dtype {
        DType::F16 => -65504.0,
        DType::BF16 => -1e38,
        _ => f32::MIN as f64,
    };
    (inverted_mask * min_value)?.to_dtype(dtype)
}

#[cfg(feature = "cuda")]
fn cumulative_seqlens(
    valid_lens: &[usize],
    device: &Device,
) -> Result<(Tensor, usize)> {
    let mut seqlens = Vec::with_capacity(valid_lens.len() + 1);
    seqlens.push(0u32);
    let mut total = 0u32;
    let mut max_seq_len = 0usize;
    for &len in valid_lens {
        total += len as u32;
        seqlens.push(total);
        max_seq_len = max_seq_len.max(len);
    }
    Ok((
        Tensor::from_vec(seqlens, valid_lens.len() + 1, device)?,
        max_seq_len.max(1),
    ))
}

#[cfg(feature = "cuda")]
fn packed_position_ids(
    valid_lens: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let total_tokens = valid_lens.iter().sum();
    let mut positions = Vec::with_capacity(total_tokens);
    for &len in valid_lens {
        positions.extend(0..len as u32);
    }
    Tensor::from_vec(positions, total_tokens, device)
}

#[cfg(feature = "cuda")]
fn pack_varlen_bsd(xs: &Tensor, valid_lens: &[usize]) -> Result<Tensor> {
    let mut packed = Vec::with_capacity(valid_lens.len());
    for (batch_idx, &len) in valid_lens.iter().enumerate() {
        packed.push(xs.i(batch_idx)?.narrow(0, 0, len.max(1))?);
    }
    Tensor::cat(&packed, 0)
}

/// Packs `(batch, seq, heads, head_dim)` rows into flash varlen's
/// `(total_tokens, heads, head_dim)` layout, keeping each sequence's
/// valid prefix only.
#[cfg(feature = "cuda")]
fn pack_varlen_thd(xs: &Tensor, valid_lens: &[usize]) -> Result<Tensor> {
    let mut packed = Vec::with_capacity(valid_lens.len());
    for (batch_idx, &len) in valid_lens.iter().enumerate() {
        packed.push(xs.i(batch_idx)?.narrow(0, 0, len.max(1))?);
    }
    Tensor::cat(&packed, 0)
}

#[cfg(feature = "cuda")]
fn unpack_varlen_bsd(
    xs: &Tensor,
    valid_lens: &[usize],
    max_seq_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let (_, dim) = xs.dims2()?;
    let mut batches = Vec::with_capacity(valid_lens.len());
    let mut offset = 0usize;
    for &len in valid_lens {
        let valid = xs.narrow(0, offset, len.max(1))?;
        offset += len;
        if len < max_seq_len {
            let pad_len = max_seq_len - len;
            let padding = Tensor::zeros((pad_len, dim), xs.dtype(), device)?;
            batches.push(Tensor::cat(&[&valid, &padding], 0)?);
        } else {
            batches.push(valid);
        }
    }
    Tensor::stack(&batches, 0)
}

// Attention mask caused by the sliding window
fn get_local_attention_mask(
    seq_len: usize,
    max_distance: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (j as i32 - i as i32).abs() > max_distance as i32 {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

// ModernBERT backbone
#[derive(Clone)]
pub struct ModernBert {
    word_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
    local_attention_masks: Arc<LastUsedCache<usize, Tensor>>,
    #[cfg(feature = "cuda")]
    varlen_positions: Arc<LastUsedCache<Vec<usize>, Tensor>>,
}

impl ModernBert {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let norm = layer_norm_no_bias_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("embeddings.norm"),
        )?;
        let global_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.global_rope_theta,
            vb.device(),
        )?);
        let local_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let layer_uses_local_attention =
                layer_id % config.global_attn_every_n_layers != 0;
            layers.push(ModernBertLayer::load(
                vb.pp(format!("layers.{layer_id}")),
                config,
                if layer_uses_local_attention {
                    local_rotary_emb.clone()
                } else {
                    global_rotary_emb.clone()
                },
                layer_uses_local_attention,
            )?);
        }

        let final_norm = layer_norm_no_bias_fused(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_norm"),
        )?;

        Ok(Self {
            word_embeddings,
            norm,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
            local_attention_masks: Arc::new(LastUsedCache::new()),
            #[cfg(feature = "cuda")]
            varlen_positions: Arc::new(LastUsedCache::new()),
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let seq_len = xs.shape().dims()[1];
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        let attention_dtype = xs.dtype();
        let global_attention_mask =
            prepare_4d_attention_mask(mask, attention_dtype, None)?;
        let local_attention_mask =
            self.local_attention_masks.get_or_try_insert(&seq_len, || {
                get_local_attention_mask(
                    seq_len,
                    self.local_attention_size / 2,
                    xs.device(),
                )?
                .to_dtype(attention_dtype)
            })?;
        let combined_local_attention_mask =
            global_attention_mask.broadcast_add(&local_attention_mask)?;
        for layer in self.layers.iter() {
            let attention_mask = if layer.uses_local_attention {
                &combined_local_attention_mask
            } else {
                &global_attention_mask
            };
            xs = layer.forward(&xs, attention_mask)?;
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_unmasked(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        let local_window = self.local_attention_size / 2;
        let full_attention_threshold = local_window * 2 + 1;
        for layer in self.layers.iter() {
            let effective_window = if layer.uses_local_attention
                && xs.dim(1)? > full_attention_threshold
            {
                Some(local_window)
            } else {
                None
            };
            xs = layer.forward_unmasked(&xs, effective_window)?;
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }

    #[cfg(feature = "cuda")]
    fn cached_packed_positions(
        &self,
        valid_lens: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        self.varlen_positions.get_or_try_insert(valid_lens, || {
            packed_position_ids(valid_lens, device)
        })
    }

    #[cfg(feature = "cuda")]
    pub fn forward_varlen_padded(
        &self,
        xs: &Tensor,
        valid_lens: &[usize],
    ) -> Result<Tensor> {
        let xs = xs.apply(&self.word_embeddings)?;
        let (seqlens, max_seq_len) =
            cumulative_seqlens(valid_lens, xs.device())?;
        let positions =
            self.cached_packed_positions(valid_lens, xs.device())?;
        let local_window = self.local_attention_size / 2;
        let full_attention_threshold = local_window * 2 + 1;
        let mut packed_xs =
            pack_varlen_bsd(&xs, valid_lens)?.apply(&self.norm)?;
        for layer in self.layers.iter() {
            let effective_window = if layer.uses_local_attention
                && max_seq_len > full_attention_threshold
            {
                Some(local_window)
            } else {
                None
            };
            packed_xs = layer.forward_varlen_packed_input(
                &packed_xs,
                valid_lens,
                &positions,
                &seqlens,
                max_seq_len,
                effective_window,
            )?;
        }
        let xs = unpack_varlen_bsd(
            &packed_xs,
            valid_lens,
            max_seq_len,
            xs.device(),
        )?;
        xs.apply(&self.final_norm)
    }

    /// Forward pass for fixed-length query batches where padding rows
    /// must produce outputs (ColBERT query expansion feeds [MASK]
    /// expansion rows into MaxSim) without serving as attention keys.
    /// Rows keep the padded layout end to end; only k/v are packed to
    /// each sequence's valid prefix per layer.
    #[cfg(feature = "cuda")]
    pub fn forward_query_varlen(
        &self,
        xs: &Tensor,
        valid_lens: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len) = xs.dims2()?;
        let local_window = self.local_attention_size / 2;
        let full_attention_threshold = local_window * 2 + 1;
        // Flash varlen aligns sliding windows to the bottom-right
        // diagonal when a sequence's key prefix is shorter than its
        // query rows, which diverges from ModernBERT's centered band.
        // No known ColBERT config pads queries past the threshold, so
        // refuse rather than silently mis-attend.
        if seq_len > full_attention_threshold
            && valid_lens.iter().any(|&len| len < seq_len)
        {
            candle_core::bail!(
                "queries longer than {full_attention_threshold} tokens \
                 with expansion padding must use the masked path"
            );
        }
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        let uniform_q = vec![seq_len; batch];
        let (seqlens_q, _) = cumulative_seqlens(&uniform_q, xs.device())?;
        let (seqlens_k, max_seqlen_k) =
            cumulative_seqlens(valid_lens, xs.device())?;
        for layer in self.layers.iter() {
            let effective_window = if layer.uses_local_attention
                && seq_len > full_attention_threshold
            {
                Some(local_window)
            } else {
                None
            };
            xs = layer.forward_query_varlen(
                &xs,
                valid_lens,
                &seqlens_q,
                &seqlens_k,
                max_seqlen_k,
                effective_window,
            )?;
        }
        xs.apply(&self.final_norm)
    }
}

// ModernBERT for the fill-mask task
#[derive(Clone)]
pub struct ModernBertForMaskedLM {
    model: ModernBert,
    decoder: ModernBertDecoder,
    head: ModernBertHead,
}

impl ModernBertForMaskedLM {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;
        let decoder = ModernBertDecoder::load(vb.clone(), config)?;
        let head = ModernBertHead::load(vb.pp("head"), config)?;
        Ok(Self {
            model,
            decoder,
            head,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let xs = self
            .model
            .forward(xs, mask)?
            .apply(&self.head)?
            .apply(&self.decoder)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertClassifier {
    classifier: Linear,
}

impl ModernBertClassifier {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // The decoder weights are tied with the embeddings layer weights
        let classifier = linear(
            config.hidden_size,
            config
                .classifier_config
                .as_ref()
                .map(|cc| cc.id2label.len())
                .unwrap_or_default(),
            vb.pp("classifier"),
        )?;
        Ok(Self { classifier })
    }
}

impl Module for ModernBertClassifier {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.classifier)?;
        softmax(&xs, D::Minus1)
    }
}

#[derive(Clone)]
pub struct ModernBertForSequenceClassification {
    model: ModernBert,
    head: ModernBertHead,
    classifier: ModernBertClassifier,
    classifier_pooling: ClassifierPooling,
}

impl ModernBertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;
        let classifier = ModernBertClassifier::load(vb.clone(), config)?;
        let head = ModernBertHead::load(vb.pp("head"), config)?;
        Ok(Self {
            model,
            head,
            classifier,
            classifier_pooling: config
                .classifier_config
                .as_ref()
                .map(|cc| cc.classifier_pooling)
                .unwrap_or_default(),
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let output = self.model.forward(xs, mask)?;
        let last_hidden_state = match self.classifier_pooling {
            ClassifierPooling::CLS => output.i((.., .., 0))?,
            ClassifierPooling::MEAN => {
                let unsqueezed_mask =
                    &mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
                let sum_output =
                    output.broadcast_mul(unsqueezed_mask)?.sum(1)?;
                sum_output.broadcast_div(
                    &mask.sum_keepdim(1)?.to_dtype(DType::F32)?,
                )?
            }
        };
        let xs = self
            .head
            .forward(&last_hidden_state)?
            .apply(&self.classifier)?;
        Ok(xs)
    }
}
