# Legs versus shadows — observation only

This note is excluded from the active benchmark request files and judging budget.
The user reported `Sanjac:0044`, `Sanjac:0045` and `Sanjac:0020` as misleading results for “legs close up”.
Inspection found real foreground legs in some frames and prominent cast shadows in others.
`Sanjac:0045` covers 41–45 seconds and overlaps the other retrieval windows; it is not an L1 label segment.

- [Native front frame #1274](../../../labels/.work/renders/l1-Sanjac/frame_Sanjac_front_1274.jpg): nearby physical legs and shadows coexist.
- [Native front frame #1349](../../../labels/.work/renders/query-expansion-shadow-review/frame_Sanjac_front_1349.jpg): shadows dominate the foreground.

A future retrieval regression should judge the selected frame separately from the containing window.
Do not automatically mark all three windows negative, or use person count as a proxy for close-up composition.
The four proposed shadow query variants were removed from the active additions to focus judging on robot behavior.
