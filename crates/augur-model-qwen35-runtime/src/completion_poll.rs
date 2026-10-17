//! A pending-query budget belongs to one submission, never to prior work.

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SubmissionPollError<E> {
    InvalidBudget,
    Query(E),
    Exhausted { pending_polls: u64, limit: u64 },
}

pub(crate) fn poll_submission<T, E>(
    limit: u64,
    mut query: impl FnMut() -> Result<Option<T>, E>,
    mut backoff: impl FnMut(u64),
) -> Result<(T, u64), SubmissionPollError<E>> {
    if limit == 0 {
        return Err(SubmissionPollError::InvalidBudget);
    }
    let mut pending_polls = 0_u64;
    loop {
        if let Some(proof) = query().map_err(SubmissionPollError::Query)? {
            return Ok((proof, pending_polls));
        }
        pending_polls += 1;
        if pending_polls >= limit {
            return Err(SubmissionPollError::Exhausted {
                pending_polls,
                limit,
            });
        }
        backoff(pending_polls);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_prefill_does_not_consume_decode_submission_budget() {
        for prefill_polls in [0, 2, 3, 462_506_845, 1_500_000_000] {
            let mut queries = 0;
            let mut waited = Vec::new();
            let (proof, decode_polls) = poll_submission(
                3,
                || {
                    queries += 1;
                    Ok::<_, ()>((queries == 3).then_some("exact_submission_proof"))
                },
                |n| waited.push(n),
            )
            .unwrap();
            assert_eq!(proof, "exact_submission_proof");
            assert_eq!(decode_polls, 2);
            assert_eq!(waited, [1, 2]);
            assert_eq!(prefill_polls + decode_polls, prefill_polls + 2);
        }
    }

    #[test]
    fn every_submission_starts_with_a_fresh_pending_budget() {
        for _ in 0..3 {
            let mut queries = 0;
            let (_, pending) = poll_submission(
                2,
                || {
                    queries += 1;
                    Ok::<_, ()>((queries == 2).then_some(()))
                },
                |_| {},
            )
            .unwrap();
            assert_eq!(pending, 1);
        }
    }

    #[test]
    fn permanently_pending_still_fails_at_the_unchanged_limit() {
        let mut queries = 0;
        let mut waited = 0;
        let result = poll_submission(
            3,
            || {
                queries += 1;
                Ok::<Option<()>, ()>(None)
            },
            |_| waited += 1,
        );
        assert_eq!(
            result,
            Err(SubmissionPollError::Exhausted {
                pending_polls: 3,
                limit: 3
            })
        );
        assert_eq!(queries, 3);
        assert_eq!(waited, 2);
    }

    #[test]
    fn ready_on_first_query_does_not_spend_or_wait() {
        let result = poll_submission(1, || Ok::<_, ()>(Some(42)), |_| panic!("no wait"));
        assert_eq!(result, Ok((42, 0)));
    }

    #[test]
    fn query_error_propagates_without_retry() {
        let result = poll_submission::<(), _>(
            3,
            || Err("foreign_or_stale_submission"),
            |_| panic!("must not hide query failure"),
        );
        assert_eq!(
            result,
            Err(SubmissionPollError::Query("foreign_or_stale_submission"))
        );
    }

    #[test]
    fn query_error_after_pending_is_not_a_completion() {
        let mut queries = 0;
        let result = poll_submission::<(), _>(
            3,
            || {
                queries += 1;
                if queries == 1 {
                    Ok(None)
                } else {
                    Err("device_error")
                }
            },
            |_| {},
        );
        assert_eq!(queries, 2);
        assert_eq!(result, Err(SubmissionPollError::Query("device_error")));
    }

    #[test]
    fn zero_budget_rejects_before_querying_the_device() {
        let result =
            poll_submission::<(), ()>(0, || panic!("not queried"), |_| panic!("not waited"));
        assert_eq!(result, Err(SubmissionPollError::InvalidBudget));
    }
}
