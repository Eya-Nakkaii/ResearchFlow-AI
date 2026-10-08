
import unittest

from quality_checks import check_numeric_consistency


SOURCES = [
    {
        "source": "rapport_fictif.pdf",
        "content": (
            "M05 compte 76 alertes. "
            "M18 compte 85 alertes. "
            "Total de 720 alertes pour l'usine."
        )
    }
]


class TestGuardExtended(unittest.TestCase):

    def test_correct_sum(self):

        result = check_numeric_consistency(
            "M05 et M18 cumulent "
            "161 alertes (76 + 85).",
            SOURCES
        )

        self.assertEqual(result.issues, [])
        self.assertGreaterEqual(
            result.checked_claims, 1
        )

    def test_wrong_sum(self):

        result = check_numeric_consistency(
            "M05 et M18 cumulent "
            "161 alertes (76 + 84).",
            SOURCES
        )

        self.assertTrue(
            any(
                "Somme incorrecte" in issue
                for issue in result.issues
            )
        )

    def test_majority_of_alerts(self):

        answer = (
            "Elles représentent 22,36 % "
            "de toutes les alertes. "
            "Elles génèrent la majorité "
            "des alertes."
        )

        result = check_numeric_consistency(
            answer,
            SOURCES
        )

        self.assertTrue(
            any(
                "Majorité contradictoire" in issue
                for issue in result.issues
            )
        )

    def test_majority_of_anomalies_not_proven(self):

        answer = (
            "M05 et M18 cumulent 161 alertes "
            "(76 + 85), soit 22,36 % "
            "de toutes les alertes. "
            "Elles génèrent la majorité "
            "des anomalies."
        )

        result = check_numeric_consistency(
            answer,
            SOURCES
        )

        self.assertTrue(
            any(
                "Majorité non justifiée" in issue
                for issue in result.issues
            )
        )

    def test_actual_writer_markdown(self):

        answer = (
            "**Volume d'alertes** : M05 et M18 "
            "cumulent **161 alertes** "
            "(76 + 85), soit **22,36 % "
            "de toutes les alertes**.\n"
            "Ces éléments expliquent la priorité : "
            "elles génèrent la majorité "
            "des anomalies."
        )

        result = check_numeric_consistency(
            answer,
            SOURCES
        )

        self.assertGreaterEqual(
            result.checked_claims, 2
        )

        self.assertTrue(
            any(
                "Majorité non justifiée" in issue
                for issue in result.issues
            )
        )

    def test_no_wrong_majority_when_over_half(self):

        answer = (
            "Elles représentent 55 % "
            "de toutes les alertes. "
            "Elles génèrent la majorité "
            "des alertes."
        )

        result = check_numeric_consistency(
            answer,
            SOURCES
        )

        self.assertEqual(result.issues, [])


if __name__ == "__main__":
    unittest.main()
