# Error Taxonomy (Section 21)

Categories to classify and count against the leakage-safe validation split
(fill in `reports/hard_cases.tsv` with representative examples per category
once real challenge data is available):

1. False merges caused by common business names
2. False merges caused by common addresses
3. False merges caused by weak country evidence
4. Missed matches caused by transliteration
5. Missed matches caused by partial addresses
6. Missed matches caused by severe typos
7. Underblocking (true match never reaches the candidate set)
8. Overblocking followed by false-positive scoring
9. Singleton false positives
10. Multi-match underprediction
11. Cross-source disagreement (S2 and S3 evidence point different ways)
12. Unfamiliar country patterns (France, unseen at training time)

For each category, compute: count, rate, representative examples, and the
pipeline component responsible (blocking pass, feature family, decision
threshold, etc.) -- this is what makes the ablation plan actionable rather
than just a table of aggregate scores.
