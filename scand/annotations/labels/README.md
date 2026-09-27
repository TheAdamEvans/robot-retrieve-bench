# Label revisions

`precedence.json` assigns priorities to shards. Higher values supersede lower values for the same label key;
unlisted shards have priority 0. The initial Butler smoke labels have priority -1, and L2 follow-up shards have
priority 1. Filesystem timestamps have no effect.

New shards can add labels without a manifest change. To revise a label from a different shard, give the revision
an explicit higher priority. Conflicting records at the same priority fail with an error. Within a single shard,
the last record for a key wins. Training, evaluation and `scandq label get` use the same reader.
