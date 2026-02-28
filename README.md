# BitcoinGoogleTrendsIndex

PLEASE DO NOT RUN THE CODE AT ALL UNTIL YOU READ THIS.
Case A: The "Exploitable Pattern" (Unidirectional: Trends -> Returns)
Condition: The p-value for Trends -> Returns is significant (< 0.05), but the p-value for Returns -> Trends is not.
Conclusion: A meaningful, one-way information flow from search interest to price returns exists.
This suggests search interest has predictive value for future returns.
Case B: Market Activity Drives Attention (Unidirectional: Returns -> Trends)
Condition: The p-value for Returns -> Trends is significant (< 0.05), but the p-value for Trends -> Returns is not.
Conclusion: Price changes are driving public attention, but public attention is not providing a predictive signal for price.
Case C: A Two-Way Relationship (Bidirectional)
Condition: Both p-values (Trends -> Returns and Returns -> Trends) are significant (< 0.05).
Action: Compare the two Transfer Entropy (TE) scores. The direction with the higher TE score is the stronger, more dominant flow.
Case D: No Pattern Found
Condition: Neither p-value is significant.
Conclusion: There is no evidence of a statistically significant predictive relationship in either direction.
