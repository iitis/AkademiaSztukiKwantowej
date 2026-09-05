#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generuje stronę WWW z materiałami (katalog `strona/` + `index.html`)
oraz pliki README.md dla katalogu `materialy/` i każdego bloku.

Uruchomienie z katalogu głównego repozytorium:

    python3 narzedzia/zbuduj_strone.py

Wymaga pakietu `nbconvert` (podglądy notatników) i programu `pdfinfo`
z pakietu poppler-utils (liczba stron w plikach PDF).
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opisy import BLOKI, OPISY_KATALOGOW, TYTULY  # noqa: E402

KORZEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATERIALY = os.path.join(KORZEN, "materialy")
STRONA = os.path.join(KORZEN, "strona")
PODGLAD = os.path.join(STRONA, "podglad")
REPO_URL = "https://github.com/iitis/AkademiaSztukiKwantowej"
POMIN_PODGLAD = ("zrodla-latex",)


# --------------------------------------------------------------------------- #
# pomocnicze
# --------------------------------------------------------------------------- #
def rozmiar(bajty: int) -> str:
    if bajty >= 1024 ** 2:
        return f"{bajty / 1024 ** 2:.1f} MB".replace(".", ",")
    return f"{max(1, round(bajty / 1024))} kB"


def strony_pdf(sciezka: str) -> int | None:
    try:
        wynik = subprocess.run(["pdfinfo", sciezka], capture_output=True, text=True, timeout=60)
        m = re.search(r"^Pages:\s+(\d+)", wynik.stdout, re.M)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def tytul_notatnika(sciezka: str) -> tuple[str, int]:
    try:
        nb = json.load(open(sciezka, encoding="utf-8"))
    except Exception:
        return "", 0
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            m = re.search(r"^#+\s*(.+)", "".join(c.get("source", [])), re.M)
            if m:
                return m.group(1).strip(), len(nb.get("cells", []))
    return "", len(nb.get("cells", []))


def czytelna_nazwa(nazwa: str) -> str:
    baza = os.path.splitext(nazwa)[0]
    baza = re.sub(r"^\d+[-_]", "", baza)
    baza = baza.replace("_", " ").replace("-", " ")
    return baza[:1].upper() + baza[1:]


# --------------------------------------------------------------------------- #
# inwentaryzacja
# --------------------------------------------------------------------------- #
def zbierz(blok: dict) -> dict:
    katalog = os.path.join(MATERIALY, blok["katalog"])
    pozycje = {"wyklady": [], "notatniki": [], "kod": [], "dane": [], "inne": [], "plan": []}
    liczby = {"pliki": 0, "bajty": 0}
    for root, dirs, files in os.walk(katalog):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, katalog)
            b = os.path.getsize(p)
            liczby["pliki"] += 1
            liczby["bajty"] += b
            wpis = dict(rel=rel, nazwa=f, bajty=b, opis=TYTULY.get(rel, ""))
            if rel.split(os.sep)[0] in POMIN_PODGLAD:
                pozycje["inne"].append(wpis)
            elif f.endswith(".pdf") and (rel.startswith("wyklady") or rel.startswith("materialy-zrodlowe")):
                wpis["strony"] = strony_pdf(p)
                pozycje["wyklady"].append(wpis)
            elif f.endswith(".ipynb"):
                wpis["tytul"], wpis["komorki"] = tytul_notatnika(p)
                wpis["tytul"] = TYTULY.get(rel) or wpis["tytul"]
                pozycje["notatniki"].append(wpis)
            elif f.endswith(".py") or f.endswith(".qasm"):
                pozycje["kod"].append(wpis)
            elif rel.startswith("plan"):
                if f.endswith(".pdf"):
                    wpis["strony"] = strony_pdf(p)
                pozycje["plan"].append(wpis)
            elif f.endswith((".npy", ".npz", ".csv", ".txt", ".gz", ".pkl", ".pt", ".safetensors",
                             "-ubyte", ".json")):
                pozycje["dane"].append(wpis)
            else:
                pozycje["inne"].append(wpis)
    return {"pozycje": pozycje, "liczby": liczby}


# --------------------------------------------------------------------------- #
# podglądy notatników
# --------------------------------------------------------------------------- #
def naprawiony_notatnik(sciezka: str) -> str | None:
    """Zwraca ścieżkę tymczasowej kopii notatnika z usuniętymi metadanymi
    widgetów (niekompletny wpis `metadata.widgets` psuje renderowanie)."""
    import tempfile
    try:
        nb = json.load(open(sciezka, encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(nb.get("metadata", {}).get("widgets"), dict):
        return None
    nb["metadata"].pop("widgets", None)
    uchwyt = tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False, encoding="utf-8")
    json.dump(nb, uchwyt, ensure_ascii=False)
    uchwyt.close()
    return uchwyt.name


def zbuduj_podglady(blok: dict, dane: dict) -> None:
    katalog = os.path.join(MATERIALY, blok["katalog"])
    for wpis in dane["pozycje"]["notatniki"]:
        zrodlo = os.path.join(katalog, wpis["rel"])
        cel = os.path.join(PODGLAD, blok["katalog"], os.path.splitext(wpis["rel"])[0] + ".html")
        os.makedirs(os.path.dirname(cel), exist_ok=True)
        polecenie = [sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images",
                     "--log-level", "ERROR", "--output", os.path.abspath(cel), zrodlo]
        wynik = subprocess.run(polecenie, capture_output=True, text=True)
        if wynik.returncode != 0 or not os.path.exists(cel):
            # Najczęstsza przyczyna: uszkodzony wpis metadata.widgets, który
            # uniemożliwia wyświetlenie notatnika także w serwisie GitHub.
            zastepczy = naprawiony_notatnik(zrodlo)
            if zastepczy:
                polecenie[-1] = zastepczy
                wynik = subprocess.run(polecenie, capture_output=True, text=True)
                os.unlink(zastepczy)
        if wynik.returncode != 0 or not os.path.exists(cel):
            print("   BŁĄD podglądu:", wpis["rel"], wynik.stderr.strip()[:200])
            wpis["podglad"] = None
            continue
        wpis["podglad"] = os.path.relpath(cel, STRONA).replace(os.sep, "/")
        if not dopisz_makra(cel):
            print("   UWAGA: nie rozpoznano konfiguracji MathJax w", wpis["rel"])
        dopisz_pasek(cel, blok, wpis)


# Notatniki używają makr \bm i \ket z pakietów bm i physics. MathJax ładowany
# przez nbconvert ich nie zna i wypisuje je jako surowy tekst, więc dokładamy
# definicje do konfiguracji generowanej przez szablon.
MAKRA_TEX = """                TeX: {
                    Macros: {
                        bm: ["{\\\\boldsymbol{#1}}", 1],
                        ket: ["{\\\\left|#1\\\\right\\\\rangle}", 1],
                        bra: ["{\\\\left\\\\langle#1\\\\right|}", 1]
                    },
"""
WZORZEC_TEX = """                TeX: {
"""


def dopisz_makra(plik: str) -> bool:
    """Wstawia definicje makr do konfiguracji MathJax w podglądzie."""
    tresc = open(plik, encoding="utf-8").read()
    if "Macros:" in tresc:
        return True
    if WZORZEC_TEX not in tresc:
        return False
    open(plik, "w", encoding="utf-8").write(tresc.replace(WZORZEC_TEX, MAKRA_TEX, 1))
    return True


PASEK = """<div style="font:14px/1.5 system-ui,sans-serif;background:#f0f3f7;border-bottom:1px solid #c8d0da;padding:10px 16px;color:#1c2530">
<a href="{powrot}" style="color:#0b4f9e">&larr; Materiały bloku: {tytul}</a>
&nbsp;·&nbsp; podgląd notatnika <code>{nazwa}</code>
&nbsp;·&nbsp; <a href="{pobierz}" style="color:#0b4f9e">pobierz plik .ipynb</a>
</div>
"""


def dopisz_pasek(plik: str, blok: dict, wpis: dict) -> None:
    glebokosc = wpis["podglad"].count("/")
    powrot = "../" * glebokosc + f"blok-{blok['katalog'][:2]}.html"
    pobierz = "../" * (glebokosc + 1) + f"materialy/{blok['katalog']}/{wpis['rel']}"
    tresc = open(plik, encoding="utf-8").read()
    pasek = PASEK.format(powrot=html.escape(powrot), tytul=html.escape(blok["tytul"]),
                         nazwa=html.escape(wpis["nazwa"]), pobierz=html.escape(pobierz))
    tresc = tresc.replace("<body", pasek.join(["", "<body"]), 1) if "<body" not in tresc else \
        re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + pasek, tresc, count=1)
    open(plik, "w", encoding="utf-8").write(tresc)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
STYL = """
:root{--tlo:#ffffff;--tekst:#15202b;--przygaszony:#4a5764;--linia:#d6dde5;
      --akcent:#0b4f9e;--akcent-tlo:#eef4fb;--karta:#f7f9fc}
@media (prefers-color-scheme:dark){
 :root{--tlo:#11161c;--tekst:#e8edf2;--przygaszony:#a8b4c0;--linia:#2b3641;
       --akcent:#7fb4f0;--akcent-tlo:#16202b;--karta:#171e26}}
*{box-sizing:border-box}
body{margin:0;background:var(--tlo);color:var(--tekst);
     font:17px/1.6 "Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif}
.pominiecie{position:absolute;left:-9999px;top:0}
.pominiecie:focus{left:8px;top:8px;background:var(--akcent);color:#fff;padding:8px 14px;z-index:9}
header.glowny{background:var(--akcent-tlo);border-bottom:1px solid var(--linia);padding:28px 0}
.srodek{max-width:960px;margin:0 auto;padding:0 20px}
h1{font-size:2rem;line-height:1.25;margin:0 0 8px}
h2{font-size:1.4rem;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--linia)}
h3{font-size:1.12rem;margin:26px 0 8px}
p{margin:0 0 14px}
a{color:var(--akcent)}
a:focus-visible,button:focus-visible{outline:3px solid var(--akcent);outline-offset:2px}
.podtytul{color:var(--przygaszony);margin:0}
.karta{background:var(--karta);border:1px solid var(--linia);border-radius:10px;
       padding:18px 20px;margin:0 0 16px}
.karta h3{margin-top:0}
.karta .meta{color:var(--przygaszony);font-size:.92rem}
table{border-collapse:collapse;width:100%;margin:0 0 18px;font-size:.96rem}
caption{text-align:left;color:var(--przygaszony);font-size:.92rem;padding-bottom:6px}
th,td{border-bottom:1px solid var(--linia);padding:8px 10px;text-align:left;vertical-align:top}
th{background:var(--karta);font-weight:600}
td.liczba{text-align:right;white-space:nowrap;color:var(--przygaszony)}
td.bez-lamania{white-space:nowrap}
.przewijane{overflow-x:auto}
code{font-family:ui-monospace,"Cascadia Code","Consolas",monospace;font-size:.92em;
     background:var(--karta);padding:1px 5px;border-radius:4px}
pre{background:var(--karta);border:1px solid var(--linia);border-radius:8px;padding:14px;
    overflow-x:auto}
pre code{background:none;padding:0}
ul{margin:0 0 14px;padding-left:22px}
li{margin-bottom:6px}
.kroki{list-style:none;padding:0;counter-reset:krok}
.kroki li{counter-increment:krok;position:relative;padding-left:44px;margin-bottom:16px}
.kroki li::before{content:counter(krok);position:absolute;left:0;top:0;width:30px;height:30px;
  border-radius:50%;background:var(--akcent);color:#fff;display:flex;align-items:center;
  justify-content:center;font-weight:700}
footer{margin-top:56px;border-top:1px solid var(--linia);background:var(--karta);
       padding:24px 0;color:var(--przygaszony);font-size:.92rem}
.nawigacja{font-size:.95rem;color:var(--przygaszony);margin-bottom:10px}
@media (max-width:620px){body{font-size:16px}h1{font-size:1.55rem}}
"""

SZKIELET = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{tytul}</title>
<meta name="description" content="{opis}">
<link rel="stylesheet" href="{prefiks}styl.css">
</head>
<body>
<a class="pominiecie" href="#tresc">Przejdź do treści</a>
<header class="glowny"><div class="srodek">
{naglowek}
</div></header>
<main id="tresc" class="srodek">
{tresc}
</main>
<footer><div class="srodek">
<p><strong>Akademia Sztuki Kwantowej</strong> — projekt Instytutu Informatyki Teoretycznej
i Stosowanej Polskiej Akademii Nauk w Gliwicach, nr NdS-II/SP/0222/2024/01.</p>
<p>Projekt dofinansowany ze środków budżetu państwa w ramach programu Ministra Nauki
i Szkolnictwa Wyższego „Nauka dla Społeczeństwa II”. Kwota dofinansowania: 1 000 000 zł.</p>
<p>Materiały udostępniane na licencji Apache 2.0. Strona projektu:
<a href="https://ask.iitis.pl/">ask.iitis.pl</a> ·
Repozytorium: <a href="{repo}">github.com/iitis/AkademiaSztukiKwantowej</a></p>
</div></footer>
</body>
</html>
"""


def strona(nazwa_pliku: str, tytul: str, opis: str, naglowek: str, tresc: str) -> None:
    sciezka = os.path.join(STRONA, nazwa_pliku) if nazwa_pliku != "index.html" \
        else os.path.join(KORZEN, "index.html")
    prefiks = "strona/" if nazwa_pliku == "index.html" else ""
    open(sciezka, "w", encoding="utf-8").write(SZKIELET.format(
        tytul=html.escape(tytul), opis=html.escape(opis), naglowek=naglowek,
        tresc=tresc, prefiks=prefiks, repo=REPO_URL))


def tabela(naglowki, wiersze, podpis=""):
    if not wiersze:
        return ""
    th = "".join(f"<th scope=\"col\">{h}</th>" for h in naglowki)
    trs = "\n".join("<tr>" + "".join(w) + "</tr>" for w in wiersze)
    cap = f"<caption>{podpis}</caption>" if podpis else ""
    return f'<div class="przewijane"><table>{cap}<thead><tr>{th}</tr></thead><tbody>\n{trs}\n</tbody></table></div>'


def strona_bloku(blok: dict, dane: dict) -> None:
    nr = blok["katalog"][:2]
    kat = f"../materialy/{blok['katalog']}"
    poz = dane["pozycje"]
    czesci = []

    czesci.append(f"<p>{html.escape(blok['opis'])}</p>")

    if poz["wyklady"]:
        wiersze = []
        for w in poz["wyklady"]:
            tytul = w["opis"] or czytelna_nazwa(w["nazwa"])
            link = f'{kat}/{w["rel"]}'.replace(os.sep, "/")
            wiersze.append([
                f'<td><a href="{html.escape(link)}">{html.escape(tytul)}</a><br>'
                f'<span class="meta"><code>{html.escape(w["nazwa"])}</code></span></td>',
                f'<td class="liczba">{w.get("strony") or "—"}</td>',
                f'<td class="liczba">{rozmiar(w["bajty"])}</td>'])
        czesci.append("<h2>Slajdy i skrypty (PDF)</h2>")
        czesci.append("<p>Pliki otwierają się bezpośrednio w przeglądarce — nie trzeba "
                      "niczego instalować ani zakładać konta.</p>")
        czesci.append(tabela(["Materiał", "Stron", "Rozmiar"], wiersze))

    if poz["notatniki"]:
        czesci.append("<h2>Notatniki Jupyter z ćwiczeniami</h2>")
        czesci.append("<p>Odnośnik <em>„otwórz podgląd”</em> prowadzi do gotowego "
                      "podglądu notatnika wraz z wynikami obliczeń i wykresami — "
                      "bez instalowania Pythona. Aby samodzielnie uruchomić kod, należy "
                      "pobrać plik <code>.ipynb</code> (patrz „Jak uruchomić ćwiczenia” poniżej).</p>")
        grupy: dict[str, list] = {}
        for w in poz["notatniki"]:
            grupy.setdefault(os.path.dirname(w["rel"]).replace(os.sep, "/"), []).append(w)
        for katalog_wzgl, lista in grupy.items():
            opis_kat = OPISY_KATALOGOW.get(katalog_wzgl, "")
            naglowek_grupy = opis_kat or katalog_wzgl
            czesci.append(f'<h3>{html.escape(naglowek_grupy)}</h3>')
            if opis_kat:
                czesci.append(f'<p class="meta">katalog <code>{html.escape(katalog_wzgl)}</code></p>')
            wiersze = []
            for w in lista:
                tytul = w.get("tytul") or czytelna_nazwa(w["nazwa"])
                plik = f'{kat}/{w["rel"]}'.replace(os.sep, "/")
                podglad = w.get("podglad")
                kol_podglad = (f'<a href="{html.escape(podglad)}">otwórz podgląd</a>'
                               if podglad else "—")
                wiersze.append([
                    f'<td>{html.escape(tytul)}<br><span class="meta">'
                    f'<code>{html.escape(w["nazwa"])}</code></span></td>',
                    f'<td class="bez-lamania">{kol_podglad}</td>',
                    f'<td class="bez-lamania"><a href="{html.escape(plik)}">pobierz</a></td>',
                    f'<td class="liczba">{w.get("komorki", "—")}</td>'])
            czesci.append(tabela(["Notatnik", "Podgląd", "Plik", "Komórek"], wiersze))

    if poz["plan"]:
        wiersze = []
        for w in poz["plan"]:
            link = f'{kat}/{w["rel"]}'.replace(os.sep, "/")
            wiersze.append([
                f'<td><a href="{html.escape(link)}">{html.escape(czytelna_nazwa(w["nazwa"]))}</a>'
                f'<br><span class="meta"><code>{html.escape(w["rel"])}</code></span></td>',
                f'<td class="liczba">{rozmiar(w["bajty"])}</td>'])
        czesci.append("<h2>Program szkolenia</h2>")
        czesci.append(tabela(["Dokument", "Rozmiar"], wiersze))

    if poz["kod"]:
        wiersze = [[f'<td><a href="{html.escape((kat + "/" + w["rel"]).replace(os.sep, "/"))}">'
                    f'<code>{html.escape(w["rel"])}</code></a></td>',
                    f'<td class="liczba">{rozmiar(w["bajty"])}</td>'] for w in poz["kod"]]
        czesci.append("<h2>Przykładowe programy</h2>")
        czesci.append(tabela(["Plik", "Rozmiar"], wiersze))

    req = []
    for kandydat in ("warsztaty/requirements.txt", "requirements.txt"):
        if os.path.exists(os.path.join(MATERIALY, blok["katalog"], kandydat)):
            req.append(kandydat)
    if poz["notatniki"]:
        instr = [
            "Zainstaluj Pythona w wersji 3.10 lub nowszej "
            "(<a href=\"https://www.python.org/downloads/\">python.org</a>).",
            f'Pobierz materiały bloku: przejdź do <a href="{REPO_URL}/tree/master/materialy/'
            f'{blok["katalog"]}">katalogu w repozytorium</a> i użyj przycisku '
            "<em>Code → Download ZIP</em> (pobiera całe repozytorium) lub sklonuj je poleceniem "
            "<code>git clone https://github.com/iitis/AkademiaSztukiKwantowej.git</code>.",
        ]
        if req:
            instr.append("Zainstaluj wymagane pakiety: <code>pip install -r "
                         f"materialy/{blok['katalog']}/{req[0]}</code>")
        instr.append("Uruchom notatniki: <code>jupyter notebook</code> — a następnie otwórz "
                     "wybrany plik <code>.ipynb</code>.")
        czesci.append("<h2>Jak uruchomić ćwiczenia</h2>")
        czesci.append("<ol class=\"kroki\">" + "".join(f"<li>{k}</li>" for k in instr) + "</ol>")

    galezie = ", ".join(f'<a href="{REPO_URL}/tree/{g}"><code>{html.escape(g)}</code></a>'
                        for g in blok["galaz"])
    czesci.append("<h2>Materiały źródłowe i historia prac</h2>")
    czesci.append(
        f"<p>Komplet plików roboczych, z których powstały powyższe materiały (źródła LaTeX "
        f"wraz z rysunkami, wersje robocze notatników, pełna historia zmian), pozostaje "
        f"w gałęziach autorskich repozytorium: {galezie}. Do korzystania z materiałów "
        f"nie jest potrzebny dostęp do tych gałęzi.</p>")

    naglowek = (f'<p class="nawigacja"><a href="../index.html">Materiały Akademii Sztuki '
                f'Kwantowej</a> → blok {int(nr)}</p>'
                f'<h1>{html.escape(blok["tytul"])}</h1>'
                f'<p class="podtytul">Prowadzenie: {html.escape(", ".join(blok["prowadzacy"]))}</p>')
    strona(f"blok-{nr}.html", f"{blok['tytul']} — Akademia Sztuki Kwantowej",
           blok["opis"][:150], naglowek, "\n".join(czesci))


def strona_glowna(wszystko: list) -> None:
    czesci = []
    czesci.append(
        "<p>Na tej stronie zebrano wszystkie materiały dydaktyczne powstałe w projekcie "
        "<strong>Akademia Sztuki Kwantowej</strong>. Materiały są bezpłatne, dostępne bez "
        "logowania i bez zakładania konta, na licencji Apache 2.0.</p>")

    czesci.append("<h2>Jak korzystać z materiałów</h2>")
    czesci.append(
        "<ul class=\"kroki\">"
        "<li><strong>Chcę tylko przeczytać wykład.</strong> Wybierz blok tematyczny "
        "i kliknij tytuł wykładu — plik PDF otworzy się w przeglądarce.</li>"
        "<li><strong>Chcę zobaczyć ćwiczenia z wynikami.</strong> W tabeli notatników "
        "kliknij „czytaj w przeglądarce”. Zobaczysz kod, wykresy i wyniki obliczeń bez "
        "instalowania czegokolwiek.</li>"
        "<li><strong>Chcę samodzielnie uruchomić kod.</strong> Pobierz materiały "
        f"(przycisk <em>Code → Download ZIP</em> na <a href=\"{REPO_URL}\">stronie "
        "repozytorium</a>), zainstaluj pakiety wymienione w pliku "
        "<code>requirements.txt</code> danego bloku i uruchom <code>jupyter notebook</code>. "
        "Szczegółowa instrukcja znajduje się na stronie każdego bloku.</li>"
        "</ul>")

    czesci.append("<h2>Bloki tematyczne</h2>")
    for blok, dane in wszystko:
        nr = blok["katalog"][:2]
        poz = dane["pozycje"]
        licz = []
        if poz["wyklady"]:
            stron = sum(w.get("strony") or 0 for w in poz["wyklady"])
            licz.append(f"{len(poz['wyklady'])} plików PDF ({stron} stron)")
        if poz["notatniki"]:
            licz.append(f"{len(poz['notatniki'])} notatników Jupyter")
        if poz["kod"]:
            licz.append(f"{len(poz['kod'])} programów przykładowych")
        czesci.append(
            f'<div class="karta"><h3><a href="strona/blok-{nr}.html">'
            f'{int(nr)}. {html.escape(blok["tytul"])}</a></h3>'
            f'<p>{html.escape(blok["opis"])}</p>'
            f'<p class="meta">Prowadzenie: {html.escape(", ".join(blok["prowadzacy"]))}<br>'
            f'Zawartość: {html.escape(" · ".join(licz))} · '
            f'{rozmiar(dane["liczby"]["bajty"])}</p></div>')

    czesci.append("<h2>Nagrania wykładów</h2>")
    czesci.append(
        "<p>Nagrania wideo wykładów online (40 nagrań, ok. 15 GB) są dostępne w dwóch "
        "miejscach: w systemie <a href=\"https://akademia.iitis.pl/\">akademia.iitis.pl</a> "
        "przy poszczególnych wydarzeniach oraz w repozytorium Zenodo pod trwałym "
        "identyfikatorem <a href=\"https://doi.org/10.5281/zenodo.22231106\">"
        "10.5281/zenodo.22231106</a>. Archiwum wszystkich wydarzeń projektu wraz z programami "
        "znajduje się na stronie <a href=\"https://ask.iitis.pl/archiwum/\">ask.iitis.pl/archiwum</a>.</p>")

    czesci.append("<h2>Jak cytować</h2>")
    czesci.append(
        "<p>Materiały Akademii Sztuki Kwantowej, Instytut Informatyki Teoretycznej i Stosowanej "
        "Polskiej Akademii Nauk, Gliwice 2024–2026, "
        f"<a href=\"{REPO_URL}\">github.com/iitis/AkademiaSztukiKwantowej</a>.</p>")

    naglowek = ('<h1>Akademia Sztuki Kwantowej — materiały szkoleniowe</h1>'
                '<p class="podtytul">Bezpłatne materiały z wykładów i warsztatów: slajdy PDF, '
                'notatniki Jupyter z ćwiczeniami i kod źródłowy przykładów.</p>')
    strona("index.html", "Akademia Sztuki Kwantowej — materiały szkoleniowe",
           "Bezpłatne materiały dydaktyczne z obliczeń kwantowych i uczenia maszynowego.",
           naglowek, "\n".join(czesci))


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #
def readme_bloku(blok: dict, dane: dict) -> None:
    nr = blok["katalog"][:2]
    poz = dane["pozycje"]
    L = [f"# {int(nr)}. {blok['tytul']}", "",
         blok["opis"], "",
         f"**Prowadzenie:** {', '.join(blok['prowadzacy'])}", "",
         f"Wygodniejszy w przeglądaniu spis tych materiałów, z podglądem notatników "
         f"w przeglądarce, znajduje się na stronie "
         f"[iitis.github.io/AkademiaSztukiKwantowej](https://iitis.github.io/AkademiaSztukiKwantowej/strona/blok-{nr}.html).",
         ""]
    if poz["wyklady"]:
        L += ["## Slajdy i skrypty (PDF)", "", "| Materiał | Plik | Stron |", "|---|---|---|"]
        for w in poz["wyklady"]:
            tytul = w["opis"] or czytelna_nazwa(w["nazwa"])
            L.append(f"| {tytul} | [`{w['rel']}`]({w['rel'].replace(os.sep, '/')}) | "
                     f"{w.get('strony') or '—'} |")
        L.append("")
    if poz["notatniki"]:
        L += ["## Notatniki Jupyter", "",
              "Notatniki można przeczytać w przeglądarce (bez instalacji) korzystając "
              "z podglądów na stronie projektu, albo pobrać i uruchomić lokalnie.", "",
              "| Notatnik | Zawartość |", "|---|---|"]
        for w in poz["notatniki"]:
            tytul = w.get("tytul") or czytelna_nazwa(w["nazwa"])
            L.append(f"| [`{w['rel']}`]({w['rel'].replace(os.sep, '/')}) | {tytul} |")
        L.append("")
    req = [k for k in ("warsztaty/requirements.txt", "requirements.txt")
           if os.path.exists(os.path.join(MATERIALY, blok["katalog"], k))]
    if poz["notatniki"]:
        L += ["## Jak uruchomić ćwiczenia", "",
              "```bash",
              "git clone https://github.com/iitis/AkademiaSztukiKwantowej.git",
              f"cd AkademiaSztukiKwantowej/materialy/{blok['katalog']}"]
        if req:
            L.append(f"pip install -r {req[0]}")
        L += ["jupyter notebook", "```", ""]
    L += ["## Materiały źródłowe", "",
          "Pliki robocze (źródła LaTeX z rysunkami, wersje robocze, historia zmian) "
          "pozostają w gałęziach autorskich: " +
          ", ".join(f"[`{g}`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/{g})"
                    for g in blok["galaz"]) + ".", "",
          "## Licencja", "",
          "Apache 2.0 — patrz plik [LICENSE](../../LICENSE) w katalogu głównym repozytorium.", ""]
    open(os.path.join(MATERIALY, blok["katalog"], "README.md"), "w",
         encoding="utf-8").write("\n".join(L))


def readme_materialow(wszystko: list) -> None:
    L = ["# Materiały szkoleniowe", "",
         "Katalog zawiera komplet materiałów dydaktycznych projektu, podzielony na pięć "
         "bloków tematycznych. Każdy blok ma własny plik `README.md` ze spisem treści.", "",
         "| Blok | Zawartość | Rozmiar |", "|---|---|---|"]
    for blok, dane in wszystko:
        poz = dane["pozycje"]
        licz = []
        if poz["wyklady"]:
            licz.append(f"{len(poz['wyklady'])} PDF")
        if poz["notatniki"]:
            licz.append(f"{len(poz['notatniki'])} notatników")
        if poz["kod"]:
            licz.append(f"{len(poz['kod'])} programów")
        L.append(f"| [{int(blok['katalog'][:2])}. {blok['tytul']}]({blok['katalog']}/) | "
                 f"{', '.join(licz)} | {rozmiar(dane['liczby']['bajty'])} |")
    L += ["", "Spis materiałów w wygodniejszej formie, z podglądem notatników w przeglądarce, "
          "znajduje się na stronie "
          "[iitis.github.io/AkademiaSztukiKwantowej](https://iitis.github.io/AkademiaSztukiKwantowej/).", ""]
    open(os.path.join(MATERIALY, "README.md"), "w", encoding="utf-8").write("\n".join(L))


# --------------------------------------------------------------------------- #
def main() -> None:
    if os.path.isdir(STRONA):
        shutil.rmtree(STRONA)
    os.makedirs(PODGLAD, exist_ok=True)
    open(os.path.join(STRONA, "styl.css"), "w", encoding="utf-8").write(STYL)
    open(os.path.join(KORZEN, ".nojekyll"), "w").write("")

    wszystko = []
    for blok in BLOKI:
        print("blok:", blok["katalog"])
        dane = zbierz(blok)
        zbuduj_podglady(blok, dane)
        wszystko.append((blok, dane))

    for blok, dane in wszystko:
        strona_bloku(blok, dane)
        readme_bloku(blok, dane)
    strona_glowna(wszystko)
    readme_materialow(wszystko)

    manifest = [{"blok": b["katalog"], "tytul": b["tytul"],
                 "pliki": d["liczby"]["pliki"], "bajty": d["liczby"]["bajty"],
                 "pozycje": d["pozycje"]} for b, d in wszystko]
    json.dump(manifest, open(os.path.join(KORZEN, "narzedzia", "manifest.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
    print("gotowe:", sum(m["pliki"] for m in manifest), "plików")


if __name__ == "__main__":
    main()
