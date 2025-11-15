/*
 * AUTO-GENERATED — TEST STATISTICS
 * Generated: 2025-11-15T14:50:22.787Z
 * Source: C:\LFM\workspace\results\test_registry_canonical.json
 */
export const testStatistics = {
  "total": 105,
  "passing": 105,
  "failing": 0,
  "passRate": "100.0%",
  "byTier": {
    "1": {
      "total": 17,
      "passing": 17
    },
    "2": {
      "total": 25,
      "passing": 25
    },
    "3": {
      "total": 11,
      "passing": 11
    },
    "4": {
      "total": 14,
      "passing": 14
    },
    "5": {
      "total": 21,
      "passing": 21
    },
    "6": {
      "total": 12,
      "passing": 12
    },
    "7": {
      "total": 5,
      "passing": 5
    }
  },
  "generatedAt": "2025-11-15T14:50:22.787Z",
  "sourceFile": "C:\\LFM\\workspace\\results\\test_registry_canonical.json"
} as const;
export function formatPassRate(){return testStatistics.passRate+' Tests Pass';}
export function formatSummary(){return testStatistics.passing+' of '+testStatistics.total+' executed tests passing';}
