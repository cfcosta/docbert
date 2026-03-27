use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
    code: &'static str,
}

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    Conflict(String),
    Internal(String),
}

impl ApiError {
    pub fn internal(err: impl std::fmt::Display) -> Self {
        Self::Internal(err.to_string())
    }
}

impl From<docbert_core::Error> for ApiError {
    fn from(err: docbert_core::Error) -> Self {
        match &err {
            docbert_core::Error::NotFound { .. } => {
                Self::NotFound(err.to_string())
            }
            _ => Self::Internal(err.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            ApiError::BadRequest(msg) => {
                (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg)
            }
            ApiError::NotFound(msg) => {
                (StatusCode::NOT_FOUND, "NOT_FOUND", msg)
            }
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, "CONFLICT", msg),
            ApiError::Internal(msg) => {
                tracing::error!("internal error: {msg}");
                (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", msg)
            }
        };

        (
            status,
            Json(ErrorBody {
                error: message,
                code,
            }),
        )
            .into_response()
    }
}
