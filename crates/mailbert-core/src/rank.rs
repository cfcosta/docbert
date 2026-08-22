//! Put the candidates of the two legs in the order of §10.1.
//!
//! §8.1 gives the two legs: Tantivy gives the BM25 candidates, and the
//! PLAID index of docbert gives the ColBERT candidates. Neither score
//! is on the scale of the other, so mailbert fuses the two *ranks* and
//! not the two scores. This is Reciprocal Rank Fusion, and it is what
//! docbert does as well.
//!
//! §8.3 then multiplies the fused score by a decay. Mail search is
//! recency-biased: a strong match from last week is almost always
//! better than an equally strong match from 2017.
//!
//! §8.4 groups last, and not first. mailbert ranks messages, because a
//! message is what matches, and then keeps the best message of each
//! thread. A thread of 40 near-copies then takes one row and not the
//! whole first page.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
};

use crate::{
    config::DEFAULT_HALF_LIFE_DAYS,
    error::Result,
    index::{Hit, MailIndex},
    message_id::MessageId,
    threading::ThreadId,
};

/// The constant of the Reciprocal Rank Fusion of §8.1.
///
/// A large `k` flattens the difference between the top ranks, so one
/// leg cannot win on its own. 60 is the value of the paper, and the
/// value that docbert uses.
pub const RRF_K: usize = 60;

/// How many candidates each leg gives before the fusion. (§8.2)
pub const CANDIDATES: usize = 100;

/// Seconds in one day.
pub const DAY: i64 = 86_400;

/// The messages of each thread, earliest first. (§8.4)
pub type Threads = BTreeMap<ThreadId, Vec<MessageId>>;

/// What puts the rows in order. (§8.3)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Sort {
    /// The fused score, with the recency prior. The default.
    #[default]
    Best,

    /// The fused score alone. `--sort score` removes the decay.
    Score,

    /// The date, newest first. `--sort date` removes the score order.
    Date,
}

/// One row of the output of §10.1: a thread, and its best message.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The message of the thread that matched best.
    pub hit: Hit,

    /// The score that put this row here.
    pub score: f32,

    /// Where the message is in its thread, counted from 1.
    pub position: usize,

    /// How many messages the thread holds.
    pub total: usize,
}

/// What a search needs that the candidates do not carry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub sort: Sort,

    /// The half-life of the decay of §8.3, in days.
    pub half_life_days: f64,

    /// The instant that the decay counts back from.
    pub now: i64,

    /// How many rows, and therefore how many threads.
    pub limit: usize,
}

impl Options {
    /// The options of §8.3, with the defaults of the config.
    pub fn new(now: i64) -> Self {
        Self {
            sort: Sort::Best,
            half_life_days: DEFAULT_HALF_LIFE_DAYS,
            now,
            limit: crate::config::DEFAULT_COUNT,
        }
    }

    pub fn with_sort(mut self, sort: Sort) -> Self {
        self.sort = sort;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_half_life(mut self, days: f64) -> Self {
        self.half_life_days = days;
        self
    }
}

/// Fuse ranked lists of keys with Reciprocal Rank Fusion. (§8.1)
///
/// The key at position `i` of a list gives `1 / (k + i + 1)` to its
/// total. A key that one list does not hold gets nothing from it. The
/// answer is highest first, and a tie goes to the smaller key, so that
/// two runs of one search give the same order.
///
/// # Examples
///
/// ```
/// use mailbert_core::rank::{RRF_K, fuse};
///
/// let bm25 = [20u64, 10, 30];
/// let semantic = [20u64, 40];
/// let fused = fuse(&[&bm25, &semantic], RRF_K);
///
/// // 20 is first in both lists, so it is first here.
/// assert_eq!(fused[0].0, 20);
/// ```
pub fn fuse(lists: &[&[u64]], k: usize) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f32> = HashMap::new();

    for list in lists {
        for (at, key) in list.iter().enumerate() {
            let rank = (k + at + 1) as f32;

            *scores.entry(*key).or_insert(0.0) += 1.0 / rank;
        }
    }

    let mut fused: Vec<(u64, f32)> = scores.into_iter().collect();

    // The tie goes to the smaller key, so that two runs of one search
    // give one order.
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    fused
}

/// The recency prior of §8.3, as a number from 0 to 1.
///
/// A message of today keeps its score. A message that is one half-life
/// old keeps one half of it. A half-life of 0 or less removes the
/// decay, which is what `--sort score` asks for.
///
/// # Examples
///
/// ```
/// use mailbert_core::rank::{DAY, decay};
///
/// let now = 100 * DAY;
///
/// assert_eq!(decay(now, now, 180.0), 1.0);
/// assert!((decay(now - 180 * DAY, now, 180.0) - 0.5).abs() < 1e-6);
/// ```
pub fn decay(date: i64, now: i64, half_life_days: f64) -> f32 {
    if half_life_days <= 0.0 {
        return 1.0;
    }

    // Mail that the server gave a date in the future keeps its score.
    let age = (now - date).max(0) as f64 / DAY as f64;

    0.5f64.powf(age / half_life_days) as f32
}

/// Read the messages of every thread that a candidate names. (§8.4)
pub fn threads_of(index: &MailIndex, hits: &[Hit]) -> Result<Threads> {
    let wanted: BTreeSet<ThreadId> =
        hits.iter().map(|hit| hit.thread).collect();
    let mut threads = Threads::new();

    for thread in wanted {
        let members = index.thread(thread)?;

        threads.insert(thread, members.into_iter().map(|hit| hit.id).collect());
    }

    Ok(threads)
}

/// Turn the candidates of the legs into the rows of §10.1.
///
/// `legs` holds one ranked list of keys for each leg, best first.
/// `ksearch` gives one list, and `search` gives two. `hits` holds the
/// row of each key, and `threads` holds the messages of each thread.
pub fn rank(
    legs: &[&[u64]],
    hits: &BTreeMap<u64, Hit>,
    threads: &Threads,
    options: Options,
) -> Vec<Row> {
    let mut scored: Vec<(Hit, f32)> = Vec::new();

    for (key, fused) in fuse(legs, RRF_K) {
        // A leg can name a key that the caller did not read back.
        let Some(hit) = hits.get(&key) else {
            continue;
        };

        // §8.3: `--sort score` and `--sort date` remove the decay.
        let score = match options.sort {
            Sort::Best => {
                fused * decay(hit.date, options.now, options.half_life_days)
            }
            Sort::Score | Sort::Date => fused,
        };

        scored.push((hit.clone(), score));
    }

    scored.sort_by(|a, b| order(a, b, options.sort));

    let mut seen: BTreeSet<ThreadId> = BTreeSet::new();
    let mut rows: Vec<Row> = Vec::new();

    for (hit, score) in scored {
        if rows.len() >= options.limit {
            break;
        }

        // §8.4: the first message of a thread that arrives here is the
        // best message of it, because the list is already in order.
        if !seen.insert(hit.thread) {
            continue;
        }

        let members = threads.get(&hit.thread);
        let position = members
            .and_then(|ids| ids.iter().position(|id| *id == hit.id))
            .map_or(1, |at| at + 1);
        let total = members.map(Vec::len).filter(|held| *held > 0).unwrap_or(1);

        rows.push(Row {
            hit,
            score,
            position,
            total,
        });
    }

    rows
}

/// Which of two candidates comes first. (§8.3)
fn order(a: &(Hit, f32), b: &(Hit, f32), sort: Sort) -> Ordering {
    let by_date = b.0.date.cmp(&a.0.date);
    let by_key = a.0.num_id.cmp(&b.0.num_id);

    match sort {
        Sort::Date => by_date.then(by_key),
        Sort::Best | Sort::Score => {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(by_date)
                .then(by_key)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Property inventory
    //!
    //! | Property | Oracle | Why it matters |
    //! | --- | --- | --- |
    //! | `prop_fusion_rewards_a_better_pair_of_ranks` | metamorphic | §8.1 fuses ranks and not scores. A fusion that is not monotone in the ranks makes a worse candidate win. |
    //! | `prop_the_decay_never_grows_with_age` | algebraic | §8.3 says that older mail ranks lower. A decay that is not monotone would move old mail up. |
    //! | `prop_one_row_for_each_thread` | model-based | §8.4 keeps the first page clear. A second row for one thread wastes it. |
    //! | `prop_the_row_is_the_best_message_of_its_thread` | model-based | §8.4 ranks messages and then groups. A row that shows a worse message hides what matched. |
    //! | `prop_the_rows_are_in_the_order_of_the_sort` | algebraic | §8.3 gives three orders, and each must hold for the whole page. |
    //! | `prop_a_smaller_limit_is_a_prefix` | algebraic | §10.1 shows a page of a longer list. A limit that changes the order gives a different page for the same query. |

    use hegel::{TestCase, generators as gs};

    use super::*;
    use crate::{
        message::{Location, Message, SEEN},
        mime,
    };

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------

    /// The smallest budget that a Tantivy writer accepts.
    const BUDGET: usize = 15_000_000;

    /// The instant that the tests call now.
    const NOW: i64 = 1_000 * DAY;

    fn id(key: &str) -> MessageId {
        MessageId::from_message_id(&format!("<{key}@x.test>"))
            .expect("an identity")
    }

    /// One candidate, with the fields that the ranking reads.
    fn hit(key: &str, num_id: u64, thread: ThreadId, day: i64) -> Hit {
        Hit {
            score: 1.0,
            id: id(key),
            num_id,
            subject: format!("Subject {key}"),
            date: day * DAY,
            thread,
        }
    }

    /// A thread named by the key of its earliest message.
    fn thread(key: &str) -> ThreadId {
        ThreadId::from_root(id(key))
    }

    /// The corpus: thread `a` holds a1..a3, `b` holds b1 and b2, and
    /// `c` holds c1 alone.
    fn corpus() -> (BTreeMap<u64, Hit>, Threads) {
        let a = thread("a1");
        let b = thread("b1");
        let c = thread("c1");

        let rows = vec![
            hit("a1", 1, a, 0),
            hit("a2", 2, a, 1),
            hit("a3", 3, a, 2),
            hit("b1", 4, b, 10),
            hit("b2", 5, b, 11),
            hit("c1", 6, c, 20),
        ];

        let mut threads: Threads = BTreeMap::new();
        for row in &rows {
            threads.entry(row.thread).or_default().push(row.id);
        }

        let hits = rows.into_iter().map(|row| (row.num_id, row)).collect();

        (hits, threads)
    }

    /// The keys of the rows, in the order that they came back.
    fn keys(rows: &[Row]) -> Vec<u64> {
        rows.iter().map(|row| row.hit.num_id).collect()
    }

    fn options() -> Options {
        Options::new(NOW).with_limit(10)
    }

    // -----------------------------------------------------------------
    // §8.1 Fusion.
    // -----------------------------------------------------------------

    #[test]
    fn a_key_that_both_legs_find_outranks_a_key_that_one_leg_finds() {
        let bm25 = [10u64, 20, 30];
        let semantic = [20u64, 40];
        let fused = fuse(&[&bm25, &semantic], RRF_K);

        assert_eq!(fused[0].0, 20);
    }

    #[test]
    fn fusion_of_nothing_gives_nothing() {
        assert!(fuse(&[], RRF_K).is_empty());
        assert!(fuse(&[&[]], RRF_K).is_empty());
    }

    #[test]
    fn fusion_names_each_key_one_time() {
        let bm25 = [10u64, 20];
        let semantic = [20u64, 10];
        let fused = fuse(&[&bm25, &semantic], RRF_K);

        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fusion_breaks_a_tie_on_the_key() {
        // Both keys hold rank 1 of one list, so the scores are equal.
        let first = [7u64];
        let second = [3u64];
        let fused = fuse(&[&first, &second], RRF_K);

        assert_eq!(
            fused.iter().map(|(key, _)| *key).collect::<Vec<u64>>(),
            vec![3, 7]
        );
    }

    #[test]
    fn a_key_of_one_leg_keeps_the_order_of_that_leg() {
        let bm25 = [30u64, 20, 10];
        let fused = fuse(&[&bm25], RRF_K);

        assert_eq!(
            fused.iter().map(|(key, _)| *key).collect::<Vec<u64>>(),
            vec![30, 20, 10]
        );
    }

    // -----------------------------------------------------------------
    // §8.3 The recency prior.
    // -----------------------------------------------------------------

    #[test]
    fn a_message_of_today_keeps_its_score() {
        assert_eq!(decay(NOW, NOW, 180.0), 1.0);
    }

    #[test]
    fn a_message_of_one_half_life_keeps_one_half_of_its_score() {
        let old = NOW - 180 * DAY;

        assert!((decay(old, NOW, 180.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_message_of_two_half_lives_keeps_one_quarter_of_its_score() {
        let old = NOW - 360 * DAY;

        assert!((decay(old, NOW, 180.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn a_message_of_the_future_keeps_its_score() {
        assert_eq!(decay(NOW + 30 * DAY, NOW, 180.0), 1.0);
    }

    #[test]
    fn a_half_life_of_zero_removes_the_decay() {
        let old = NOW - 3_650 * DAY;

        assert_eq!(decay(old, NOW, 0.0), 1.0);
        assert_eq!(decay(old, NOW, -1.0), 1.0);
    }

    #[test]
    fn the_recency_prior_can_move_a_better_match_down() {
        let (hits, threads) = corpus();

        // c1 is 20 days old and b1 is 10 days newer than a1, so a very
        // short half-life must put the newest first.
        let legs = [1u64, 4, 6];
        let sharp = options().with_half_life(30.0);
        let rows = rank(&[&legs], &hits, &threads, sharp);

        assert_eq!(keys(&rows), vec![6, 4, 1]);
    }

    // -----------------------------------------------------------------
    // §8.3 The three orders.
    // -----------------------------------------------------------------

    #[test]
    fn sort_score_removes_the_decay() {
        let (hits, threads) = corpus();
        let legs = [1u64, 4, 6];
        let plain = options().with_sort(Sort::Score).with_half_life(30.0);
        let rows = rank(&[&legs], &hits, &threads, plain);

        assert_eq!(keys(&rows), vec![1, 4, 6]);
    }

    #[test]
    fn sort_date_gives_the_newest_first() {
        let (hits, threads) = corpus();
        let legs = [1u64, 4, 6];
        let by_date = options().with_sort(Sort::Date);
        let rows = rank(&[&legs], &hits, &threads, by_date);

        assert_eq!(keys(&rows), vec![6, 4, 1]);
    }

    #[test]
    fn sort_date_ignores_the_rank_of_the_legs() {
        let (hits, threads) = corpus();
        let forward = [1u64, 4, 6];
        let backward = [6u64, 4, 1];
        let by_date = options().with_sort(Sort::Date);

        let first = rank(&[&forward], &hits, &threads, by_date);
        let second = rank(&[&backward], &hits, &threads, by_date);

        assert_eq!(keys(&first), keys(&second));
    }

    // -----------------------------------------------------------------
    // §8.4 Thread grouping.
    // -----------------------------------------------------------------

    #[test]
    fn one_row_for_each_thread() {
        let (hits, threads) = corpus();

        // Every message of thread `a` matched.
        let legs = [1u64, 2, 3, 4];
        let rows = rank(&[&legs], &hits, &threads, options());

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].hit.thread, thread("a1"));
        assert_eq!(rows[1].hit.thread, thread("b1"));
    }

    #[test]
    fn the_row_shows_the_message_that_matched_best() {
        let (hits, threads) = corpus();

        // a3 is first, so it stands for the thread even though a1 is
        // the earliest message of it.
        let legs = [3u64, 1, 2];
        let rows = rank(&[&legs], &hits, &threads, options());

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].hit.num_id, 3);
    }

    #[test]
    fn the_row_shows_the_position_of_the_message_in_its_thread() {
        let (hits, threads) = corpus();
        let legs = [3u64, 5];
        let plain = options().with_sort(Sort::Score);
        let rows = rank(&[&legs], &hits, &threads, plain);

        assert_eq!((rows[0].position, rows[0].total), (3, 3));
        assert_eq!((rows[1].position, rows[1].total), (2, 2));
    }

    #[test]
    fn a_message_of_a_thread_that_is_not_known_stands_alone() {
        let (hits, _) = corpus();
        let empty: Threads = BTreeMap::new();
        let legs = [2u64];
        let rows = rank(&[&legs], &hits, &empty, options());

        assert_eq!((rows[0].position, rows[0].total), (1, 1));
    }

    #[test]
    fn the_limit_counts_threads_and_not_messages() {
        let (hits, threads) = corpus();
        let legs = [1u64, 2, 3, 4, 5, 6];
        let one = options().with_limit(1);
        let rows = rank(&[&legs], &hits, &threads, one);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].hit.thread, thread("a1"));
    }

    #[test]
    fn a_key_that_no_candidate_holds_gives_no_row() {
        let (hits, threads) = corpus();
        let legs = [99u64, 1];
        let rows = rank(&[&legs], &hits, &threads, options());

        assert_eq!(keys(&rows), vec![1]);
    }

    #[test]
    fn a_search_with_no_candidate_gives_no_row() {
        let (hits, threads) = corpus();

        assert!(rank(&[], &hits, &threads, options()).is_empty());
    }

    #[test]
    fn ksearch_is_the_ranking_with_one_leg() {
        let (hits, threads) = corpus();
        let bm25 = [6u64, 4, 1];
        let plain = options().with_sort(Sort::Score);

        let one = rank(&[&bm25], &hits, &threads, plain);
        let two = rank(&[&bm25, &[]], &hits, &threads, plain);

        assert_eq!(keys(&one), keys(&two));
    }

    // -----------------------------------------------------------------
    // §8.4 The threads of the index.
    // -----------------------------------------------------------------

    fn raw_bytes(key: &str, day: i64) -> Vec<u8> {
        let _ = day;

        format!(
            "From: Alice Smith <alice@example.test>\r\n\
             To: bob@example.test\r\n\
             Subject: Deposit\r\n\
             Date: Fri, 14 Aug 2026 09:30:00 +0000\r\n\
             Message-ID: <{key}@x.test>\r\n\
             \r\n\
             The deposit is due.\r\n"
        )
        .into_bytes()
    }

    fn indexed(key: &str, day: i64) -> Message {
        let raw = raw_bytes(key, day);
        let location = Location {
            account: "work".to_string(),
            folder: "INBOX".to_string(),
            uid: 1,
            uid_validity: 1,
            received: NOW,
        };

        let mut found = Message::new(
            mime::parse(&raw).expect("a message"),
            location,
            [SEEN],
        );
        found.date = day * DAY;

        found
    }

    #[test]
    fn reads_the_threads_that_the_candidates_name() {
        let index = MailIndex::open_in_ram().expect("an index");
        let mut writer = index.writer(BUDGET).expect("a writer");
        let no_tags = BTreeSet::new();

        let root = indexed("a1", 0);
        let reply = indexed("a2", 1);
        let alone = indexed("c1", 20);

        let first = ThreadId::from_root(root.id);
        let second = ThreadId::from_root(alone.id);

        for (found, of) in [(&root, first), (&reply, first), (&alone, second)] {
            index.add(&writer, found, of, &no_tags).expect("a write");
        }
        index.commit(&mut writer).expect("a commit");

        let candidates = index.thread(first).expect("a thread");
        let threads = threads_of(&index, &candidates).expect("the threads");

        assert_eq!(threads.len(), 1);
        assert_eq!(threads[&first].len(), 2);
        assert_eq!(threads[&first][0], root.id);
        assert_eq!(threads[&first][1], reply.id);
    }

    #[test]
    fn reads_no_thread_for_no_candidate() {
        let index = MailIndex::open_in_ram().expect("an index");

        assert!(threads_of(&index, &[]).expect("the threads").is_empty());
    }

    // -----------------------------------------------------------------
    // Properties.
    // -----------------------------------------------------------------

    fn a_rank() -> impl gs::Generator<usize> {
        gs::integers::<usize>().min_value(0).max_value(5)
    }

    /// A ranked list of the keys of the corpus, without a repeat.
    #[hegel::composite]
    fn a_leg(tc: TestCase) -> Vec<u64> {
        let wanted = tc.draw(gs::integers::<usize>().min_value(0).max_value(6));
        let mut keys: Vec<u64> = (1..=6).collect();
        let mut leg = Vec::with_capacity(wanted);

        for _ in 0..wanted {
            if keys.is_empty() {
                break;
            }

            let at = tc.draw(
                gs::integers::<usize>()
                    .min_value(0)
                    .max_value(keys.len() - 1),
            );
            leg.push(keys.remove(at));
        }

        leg
    }

    fn a_sort() -> impl gs::Generator<Sort> {
        gs::sampled_from(vec![Sort::Best, Sort::Score, Sort::Date])
    }

    #[hegel::test(test_cases = 200)]
    fn prop_fusion_rewards_a_better_pair_of_ranks(tc: TestCase) {
        let better = tc.draw(a_rank());
        let worse = tc.draw(a_rank());
        let (low, high) = if better <= worse {
            (better, worse)
        } else {
            (worse, better)
        };

        // Key 1 sits at or above key 2 in both lists.
        let mut first = vec![0u64; 6];
        let mut second = vec![0u64; 6];
        for (at, slot) in first.iter_mut().enumerate() {
            *slot = 100 + at as u64;
        }
        second.copy_from_slice(&first);

        first[low] = 1;
        first[high] = 2;
        second[low] = 1;
        second[high] = 2;

        if low == high {
            return;
        }

        let fused = fuse(&[&first, &second], RRF_K);
        let of = |key: u64| {
            fused
                .iter()
                .find(|(found, _)| *found == key)
                .map(|(_, score)| *score)
                .expect("the key is there")
        };

        assert!(of(1) >= of(2), "a better pair of ranks scored lower");
    }

    #[hegel::test(test_cases = 200)]
    fn prop_the_decay_never_grows_with_age(tc: TestCase) {
        let older =
            tc.draw(gs::integers::<i64>().min_value(0).max_value(4_000));
        let newer =
            tc.draw(gs::integers::<i64>().min_value(0).max_value(4_000));
        let (young, old) = if newer <= older {
            (newer, older)
        } else {
            (older, newer)
        };

        let first = decay(NOW - young * DAY, NOW, 180.0);
        let second = decay(NOW - old * DAY, NOW, 180.0);

        assert!(first >= second, "an older message decayed less");
        assert!(second > 0.0, "the decay reached zero");
        assert!(first <= 1.0, "the decay grew the score");
    }

    #[hegel::test(test_cases = 150)]
    fn prop_one_row_for_each_thread(tc: TestCase) {
        let (hits, threads) = corpus();
        let leg = tc.draw(a_leg());
        let sort = tc.draw(a_sort());
        let rows = rank(&[&leg], &hits, &threads, options().with_sort(sort));

        let mut seen: BTreeSet<ThreadId> = BTreeSet::new();
        for row in &rows {
            assert!(seen.insert(row.hit.thread), "a thread took two rows");
        }

        let wanted: BTreeSet<ThreadId> = leg
            .iter()
            .filter_map(|key| hits.get(key))
            .map(|hit| hit.thread)
            .collect();

        assert_eq!(seen, wanted, "a thread of the candidates has no row");
    }

    #[hegel::test(test_cases = 150)]
    fn prop_the_row_is_the_best_message_of_its_thread(tc: TestCase) {
        let (hits, threads) = corpus();
        let leg = tc.draw(a_leg());
        let plain = options().with_sort(Sort::Score);
        let rows = rank(&[&leg], &hits, &threads, plain);

        for row in &rows {
            let best = leg
                .iter()
                .filter_map(|key| hits.get(key))
                .find(|hit| hit.thread == row.hit.thread)
                .expect("the thread matched");

            assert_eq!(
                row.hit.num_id, best.num_id,
                "the row shows a message that ranked lower"
            );
        }
    }

    #[hegel::test(test_cases = 150)]
    fn prop_the_rows_are_in_the_order_of_the_sort(tc: TestCase) {
        let (hits, threads) = corpus();
        let leg = tc.draw(a_leg());
        let sort = tc.draw(a_sort());
        let rows = rank(&[&leg], &hits, &threads, options().with_sort(sort));

        for pair in rows.windows(2) {
            let (first, second) = (&pair[0], &pair[1]);

            match sort {
                Sort::Date => assert!(
                    first.hit.date >= second.hit.date,
                    "`--sort date` gave an older message first"
                ),
                Sort::Best | Sort::Score => assert!(
                    first.score >= second.score,
                    "a lower score came first"
                ),
            }
        }
    }

    #[hegel::test(test_cases = 150)]
    fn prop_a_smaller_limit_is_a_prefix(tc: TestCase) {
        let (hits, threads) = corpus();
        let leg = tc.draw(a_leg());
        let sort = tc.draw(a_sort());
        let short = tc.draw(gs::integers::<usize>().min_value(0).max_value(6));

        let whole = rank(&[&leg], &hits, &threads, options().with_sort(sort));
        let page = rank(
            &[&leg],
            &hits,
            &threads,
            options().with_sort(sort).with_limit(short),
        );

        assert!(page.len() <= short, "the page is longer than the limit");
        assert_eq!(keys(&page), keys(&whole)[..page.len()].to_vec());
    }
}
