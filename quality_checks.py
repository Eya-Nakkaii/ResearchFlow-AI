
import re
import unicodedata
from dataclasses import dataclass


@dataclass
class GuardReport:
    checked_claims: int
    issues: list[str]


# ==========================================
# 1. OUTILS DE NORMALISATION
# ==========================================

def normalize(text: str) -> str:

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("**", "")

    return " ".join(text.split())


def to_float(value: str) -> float:

    return float(value.replace(",", "."))


# ==========================================
# 2. FAITS EXTRAITS DES SOURCES
# ==========================================

def extract_machine_alerts(
    sources: list[dict]
) -> dict[str, set[int]]:

    facts: dict[str, set[int]] = {}

    pattern = re.compile(
        r"\b(?P<machine>M\d{2})\s+"
        r"(?:compte|poss[eè]de|totalise)\s+"
        r"(?P<count>\d+)\s+alertes\b",
        re.IGNORECASE
    )

    for source in sources:

        text = normalize(source["content"])

        for match in pattern.finditer(text):

            machine = match.group("machine").upper()
            count = int(match.group("count"))

            facts.setdefault(machine, set()).add(count)

    return facts


def extract_global_alert_total(
    sources: list[dict]
) -> int | None:

    # Ne pas confondre total d'alertes
    # et nombre de cycles ou de machines.
    patterns = [
        r"\btotal\s+(?:global\s+)?"
        r"(?:de\s+)?(\d+)\s+alertes\b",

        r"\bsur\s+(?:un\s+total\s+de\s+)?"
        r"(\d+)\s+alertes\b",

        r"\b(\d+)\s+alertes\s+"
        r"(?:au\s+total|de\s+l['’]usine|"
        r"dans\s+l['’]usine)\b",

        r"\b(\d+)\s+alertes\s+d[eé]tect[eé]es"
        r"\s+dans\s+l['’]usine\b"
    ]

    totals = set()

    for source in sources:

        text = normalize(source["content"])

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                text,
                re.IGNORECASE
            ):

                totals.add(int(match.group(1)))

    # Si plusieurs totaux incompatibles
    # sont trouvés, on ne choisit pas au hasard.
    if len(totals) != 1:
        return None

    return next(iter(totals))


# ==========================================
# 3. NUMERIC GUARD
# ==========================================

def check_numeric_consistency(
    answer: str,
    sources: list[dict]
) -> GuardReport:

    issues = []
    checked = 0

    facts = extract_machine_alerts(sources)
    global_total = extract_global_alert_total(sources)

    text = normalize(answer)

    # --------------------------------------
    # A. NOMBRE D'ALERTES PAR MACHINE
    # --------------------------------------

    machine_pattern = re.compile(
        r"\b(?P<machine>M\d{2})\s+"
        r"(?:compte|poss[eè]de|totalise)\s+"
        r"(?P<count>\d+)\s+alertes\b",
        re.IGNORECASE
    )

    for match in machine_pattern.finditer(text):

        machine = match.group("machine").upper()
        claimed = int(match.group("count"))

        expected_values = facts.get(machine, set())

        if len(expected_values) != 1:
            continue

        expected = next(iter(expected_values))
        checked += 1

        if claimed != expected:

            issues.append(
                f"{machine} : {claimed} alertes "
                f"annoncées au lieu de {expected}."
            )

    # --------------------------------------
    # B. SOMMES EXPLICITES
    # Exemple : 161 alertes (76 + 85)
    # --------------------------------------

    addition_pattern = re.compile(
        r"\b(?P<total>\d+)\s+alertes\s*"
        r"\(\s*(?P<a>\d+)\s*\+\s*"
        r"(?P<b>\d+)\s*\)",
        re.IGNORECASE
    )

    for match in addition_pattern.finditer(text):

        claimed_total = int(match.group("total"))
        a = int(match.group("a"))
        b = int(match.group("b"))

        actual_total = a + b
        checked += 1

        if claimed_total != actual_total:

            issues.append(
                f"Somme incorrecte : "
                f"{a} + {b} = {actual_total}, "
                f"pas {claimed_total}."
            )

    # --------------------------------------
    # C. POURCENTAGES D'ALERTES
    # Exemple : 76 alertes, soit 6,67 %
    # --------------------------------------

    percent_pattern = re.compile(
        r"\b(?P<count>\d+)\s+alertes\b"
        r".{0,60}?\bsoit\s+"
        r"(?P<pct>\d+(?:[.,]\d+)?)\s*%"
        r"(?:\s+(?:des|sur|de\s+toutes\s+les)"
        r"\s+(?P<total>\d+)\s+alertes)?",
        re.IGNORECASE
    )

    share_hint = re.compile(
        r"(?:ensemble\s+des\s+alertes|"
        r"toutes?\s+les\s+alertes|"
        r"alertes\s+de\s+l['’]usine|"
        r"total\s+des\s+alertes)",
        re.IGNORECASE
    )

    # Ligne par ligne : permet de gérer
    # les cellules des tableaux Markdown.
    for raw_line in answer.splitlines():

        line = normalize(raw_line)

        for match in percent_pattern.finditer(line):

            count = int(match.group("count"))
            claimed_pct = to_float(
                match.group("pct")
            )

            explicit_total = match.group("total")

            if explicit_total is not None:

                denominator = int(explicit_total)

            elif (
                global_total is not None
                and share_hint.search(line)
            ):

                denominator = global_total

            else:

                # Impossible de justifier
                # le dénominateur.
                continue

            if denominator == 0:

                issues.append(
                    "Calcul impossible : dénominateur nul."
                )
                continue

            expected_pct = 100 * count / denominator
            checked += 1

            if abs(
                claimed_pct - expected_pct
            ) > 0.02:

                issues.append(
                    f"Part d'alertes incorrecte : "
                    f"{count}/{denominator} = "
                    f"{expected_pct:.2f} %, "
                    f"pas {claimed_pct:.2f} %."
                )

    # --------------------------------------
    # D. COMPARAISON À UN SEUIL
    # --------------------------------------

    threshold_pattern = re.compile(
        r"\bsup[eé]rieur(?:s)?\s+[aà]\s+"
        r"(?P<limit>\d+(?:[.,]\d+)?)\s*%"
        r"\s+pour\s+ces\s+deux\s+machines\s*"
        r"\((?P<values>[^)]*)\)",
        re.IGNORECASE
    )

    detail_pattern = re.compile(
        r"(?P<value>\d+(?:[.,]\d+)?)\s*%"
        r"\s+pour\s+(?P<machine>M\d{2})",
        re.IGNORECASE
    )

    for match in threshold_pattern.finditer(text):

        threshold = to_float(
            match.group("limit")
        )

        for item in detail_pattern.finditer(
            match.group("values")
        ):

            value = to_float(item.group("value"))
            machine = item.group("machine").upper()

            checked += 1

            if value <= threshold:

                issues.append(
                    f"{machine} : {value:.2f} % "
                    f"n'est pas supérieur à "
                    f"{threshold:.2f} %."
                )

    # --------------------------------------
    # E. VÉRIFICATION DES MAJORITÉS
    # --------------------------------------

    global_share_pattern = re.compile(
        r"\b(?P<pct>\d+(?:[.,]\d+)?)\s*%\s+"
        r"(?:de\s+)?"
        r"(?:toutes?\s+les|"
        r"l['’]ensemble\s+des|"
        r"du\s+total\s+des)\s+"
        r"(?P<kind>alertes|anomalies)\b",
        re.IGNORECASE
    )

    majority_pattern = re.compile(
        r"\bmajorit[eé]\s+des\s+"
        r"(?P<kind>alertes|anomalies)\b",
        re.IGNORECASE
    )

    shares = list(
        global_share_pattern.finditer(text)
    )

    majorities = list(
        majority_pattern.finditer(text)
    )

    # Cette règle ne s'applique que si
    # une seule proportion globale est citée.
    if len(shares) == 1 and majorities:

        share = shares[0]
        value = to_float(share.group("pct"))

        if value < 50:

            for majority in majorities:

                distance = abs(
                    majority.start() - share.start()
                )

                # Limiter le rapprochement
                # à des affirmations voisines.
                if distance > 800:
                    continue

                checked += 1

                share_kind = (
                    share.group("kind").lower()
                )

                majority_kind = (
                    majority.group("kind").lower()
                )

                if share_kind == majority_kind:

                    issues.append(
                        f"Majorité contradictoire : "
                        f"{value:.2f} % des "
                        f"{share_kind} n'est pas "
                        f"une majorité (> 50 %)."
                    )

                else:

                    issues.append(
                        f"Majorité non justifiée : "
                        f"{value:.2f} % des "
                        f"{share_kind} ne prouve pas "
                        f"la majorité des "
                        f"{majority_kind}."
                    )

    return GuardReport(
        checked_claims=checked,
        issues=issues
    )


# ==========================================
# 4. DÉMONSTRATION
# ==========================================

if __name__ == "__main__":

    demo_sources = [
        {
            "source": "rapport.pdf",
            "content": (
                "M05 compte 76 alertes. "
                "M18 compte 85 alertes. "
                "Total de 720 alertes."
            )
        }
    ]

    demo_answer = (
        "M05 et M18 cumulent 161 alertes "
        "(76 + 85), soit 22,36 % "
        "de toutes les alertes. "
        "Elles génèrent la majorité "
        "des anomalies."
    )

    report = check_numeric_consistency(
        demo_answer,
        demo_sources
    )

    print("\n=== NUMERIC GUARD ===")
    print("Contrôles :", report.checked_claims)

    for issue in report.issues:
        print("-", issue)
