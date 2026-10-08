
import sys
import unittest

from quality_checks import check_numeric_consistency


TEST_SOURCES = [
    {
        "source": "rapport_fictif.pdf — page 6",
        "content": (
            "M05 compte 76 alertes. "
            "M18 compte 85 alertes. "
            "Les machines cumulent 161 alertes, "
            "soit 22,36 % des 720 alertes. "
            "Le total de 720 alertes correspond "
            "à l'ensemble des alertes de l'usine. "
            "M05 appartient au cluster C3 Prioritaire "
            "avec un rework de 16,50 %. "
            "M18 appartient au cluster C3 Prioritaire "
            "avec un rework de 23,75 %."
        )
    },
    {
        "source": "rapport_fictif.pdf — page 8",
        "content": (
            "M24 appartient au cluster C2 Surveillance. "
            "M05 et M18 sont prioritaires."
        )
    }
]


class TestNumericGuard(unittest.TestCase):

    def check(
        self,
        answer: str
    ):
        return check_numeric_consistency(
            answer,
            TEST_SOURCES
        )

    def test_correct_alert_counts(self):

        report = self.check(
            "M05 compte 76 alertes. "
            "M18 compte 85 alertes."
        )

        self.assertEqual(report.issues, [])
        self.assertEqual(report.checked_claims, 2)

    def test_wrong_machine_count(self):

        report = self.check(
            "M05 possède 161 alertes."
        )

        self.assertEqual(len(report.issues), 1)
        self.assertIn("M05", report.issues[0])

    def test_correct_percentage(self):

        report = self.check(
            "161 alertes, soit "
            "22,36 % des 720 alertes."
        )

        self.assertEqual(report.issues, [])
        self.assertGreaterEqual(
            report.checked_claims, 1
        )

    def test_wrong_percentage(self):

        report = self.check(
            "161 alertes, soit "
            "30 % des 720 alertes."
        )

        self.assertTrue(report.issues)

    def test_wrong_threshold(self):

        report = self.check(
            "Les taux de rework sont supérieurs "
            "à 20 % pour ces deux machines "
            "(16,50 % pour M05 et "
            "23,75 % pour M18)."
        )

        self.assertEqual(len(report.issues), 1)
        self.assertIn("M05", report.issues[0])

    def test_unchecked_statement(self):

        report = self.check(
            "M05 et M18 sont prioritaires."
        )

        self.assertEqual(report.checked_claims, 0)
        self.assertEqual(report.issues, [])

    def test_wrong_percentage_in_markdown_table(self):

        answer = (
            "| Machine | Part des alertes |\n"
            "| M05 | 76 alertes, soit 6,67 % "
            "de l'ensemble des alertes |\n"
            "| M18 | 85 alertes, soit 7,69 % "
            "de l'ensemble des alertes |"
        )

        report = self.check(answer)

        self.assertEqual(len(report.issues), 2)
        self.assertEqual(report.checked_claims, 2)

    def test_correct_percentage_in_markdown_table(self):

        answer = (
            "| Machine | Part des alertes |\n"
            "| M05 | 76 alertes, soit 10,56 % "
            "de l'ensemble des alertes |\n"
            "| M18 | 85 alertes, soit 11,81 % "
            "de l'ensemble des alertes |"
        )

        report = self.check(answer)

        self.assertEqual(report.issues, [])
        self.assertEqual(report.checked_claims, 2)

    def test_no_global_total(self):

        sources = [
            {
                "source": "rapport_partiel.pdf",
                "content": "M05 compte 76 alertes."
            }
        ]

        report = check_numeric_consistency(
            "M05 : 76 alertes, soit 6,67 % "
            "de l'ensemble des alertes.",
            sources
        )

        self.assertEqual(report.checked_claims, 0)
        self.assertEqual(report.issues, [])


# ==========================================
# BENCHMARK LLM OPTIONNEL
# ==========================================

def run_llm_benchmark():

    from evaluator import evaluate_answer

    question = (
        "Quelles machines sont prioritaires "
        "dans le rapport industriel et pourquoi ?"
    )

    cases = [
        (
            "Réponse correcte",
            True,
            "M05 et M18 sont prioritaires (C3). "
            "M05 compte 76 alertes et M18 85. "
            "Elles cumulent 161 alertes, soit "
            "22,36 % des 720 alertes [S1]. "
            "M24 est classée C2 Surveillance [S2]."
        ),
        (
            "Mauvaise attribution",
            False,
            "M05 possède 161 alertes. "
            "M18 possède 85 alertes [S1]."
        ),
        (
            "Mauvaise classification",
            False,
            "M05, M18 et M24 appartiennent "
            "au cluster C3 Prioritaire [S1][S2]."
        ),
        (
            "Comparaison incorrecte",
            False,
            "M05 et M18 sont prioritaires. "
            "Leurs taux de rework sont supérieurs "
            "à 20 % pour ces deux machines "
            "(16,50 % pour M05 et "
            "23,75 % pour M18) [S1]."
        )
    ]

    correct = 0
    evaluated = 0
    false_acceptances = 0
    false_rejections = 0

    for name, expected, answer in cases:

        print(f"\nCas : {name}")

        try:

            result = evaluate_answer(
                question,
                answer,
                TEST_SOURCES
            )

        except Exception as error:

            print(
                "Évaluation impossible :",
                type(error).__name__
            )
            continue

        evaluated += 1

        if result.is_valid == expected:
            correct += 1

        elif result.is_valid and not expected:
            false_acceptances += 1

        else:
            false_rejections += 1

        print("Attendu :", expected)
        print("Obtenu :", result.is_valid)
        print("Score :", result.score)

        for issue in result.issues:
            print("-", issue)

    print("\n=== MÉTRIQUES ===")
    print("Cas prévus :", len(cases))
    print("Cas évalués :", evaluated)
    print("Décisions correctes :", correct)
    print("Fausses acceptations :", false_acceptances)
    print("Faux rejets :", false_rejections)

    if evaluated:
        print(
            "Accuracy :",
            f"{correct / evaluated:.2%}"
        )


if __name__ == "__main__":

    if "--llm" in sys.argv:
        run_llm_benchmark()
    else:
        unittest.main()
