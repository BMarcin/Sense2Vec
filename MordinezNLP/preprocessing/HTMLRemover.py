from selectolax.parser import HTMLParser


def remove_html(html: str) -> str or None:
    ## based on https://rushter.com/blog/python-fast-html-parser/
    tree = HTMLParser(html)

    if tree.body is None:
        return None

    for tag in tree.css('script'):
        tag.decompose()
    for tag in tree.css('style'):
        tag.decompose()

    text = tree.body.text(separator=' \n')
    return text


if __name__ == '__main__':
    doc = """
    <div class="col-md-6 col-xl-8 text-justify hide" id="opis1">
<p style="text-align-last: center;">
<span style="font-weight: 500;">Zakład elektromechaniczny Karol Gmerek </span>działa na rynku od 1986 roku. Łącząc w
sobie kompetencje, tradycję i doświadczenie, od ponad 30 lat świadczymy kompleksowe usługi w
dziedzinie sprzedaży, obsługi i naprawy Państwa sprzętu. Naszą specjalnością jest przezwajanie silników, przezwajanie
hamulców elektromagnetycznych oraz remonty i przezwajanie pomp zatapialnych i pomp ściekowych.
Nasz zakład zapewnia wysoką jakość świadczonych usług, konkurencyjne ceny i dyspozycyjność terminów, dzięki
którym cieszymy się zaufaniem firm z całej Polski. Naszym głównym obszarem działań jest Inowrocław, jednak podejmujemy
się zleceń z całego kraju.
</p>
</div>

<div class="col-md-1"></div>

<div class="col-sm-12 text-center mt-3 mb-5 hide reszta">
<a href="#kontakt" title="Kontakt">
<button id="skontaktujsie" class="btn mt-4">Kontakt</button>
</a>
<div class="mt-2" style="font-weight: 500">Masz pytanie? Chętnie odpowiemy.</div>
</div>

<div class="row no-gutters text-center mt-5 mb-5 hide reszta" id="zdjecia">
<div class="col-lg-4 p-2">
<img src="img/n2.jpg" class="img-fluid" alt="Silnik elektryczny" title="Silnik elektryczny" />
</div>
<div class="col-lg-4 p-2">
<img src="img/n3.jpg" class="img-fluid" alt="Pompa wodna" title="Pompa wodna" />
</div>
<div class="col-lg-4 p-2">
<img src="img/xdd.jpg" class="img-fluid" alt="Prasa" title="Prasa" />
</div>
<div class="col-lg-12 p-2">
<img src="img/pompy.jpg" class="img-fluid" alt="Prasa" title="Pompy" />
</div>
</div>

</div>

<div class="row mt-5 mb-5 hide reszta">
<div class="col-lg-5 offset-lg-1 col-xl-4 offset-xl-2 text-justify" id="oferta">
<div class="h1">OFERTA</div>
<p>
Oferujemy szeroki zakres niżej wymienionych prac:
</p>
<ul>
<li>regeneracja pomp obiegowych,</li>
<li>próby ciśnieniowe pomp,</li>
<li>naprawa sprzęgieł i hamulców elektrycznych,</li>
<li>wykonujemy próby i pomiary elektryczne,</li>
<li>kompleksowe przeglądy oraz konserwacje silników,</li>
<li>przezwajanie wirników prądu stałego,</li>
<li>przezwajanie silników krótkozwartych,</li>
<li>przezwajanie silników pierścieniowych,</li>
<li>przezwajanie silników</li>
<li>regeneracja osi napędowej,</li>
<li>wyważanie wirnika,</li>
<li>regeneracja tarczy łożyskowej,</li>
<li>pompy zatapialne wycena indywidualna uzależniona od typu (producent pompy),</li>
<li>przezwajanie cewek elektromagnetycznych.</li>
</ul>
<p>
Naszym głównym obszarem działań jest Inowrocław, jednak podejmujemy
się zleceń z całego kraju.
</p>
</div>
<div class="col-lg-5 col-xl-4 text-justify">
<img src="img/xddd.jpg" class="img-fluid" alt="Prasa" title="Prasa" />
</div>
</div>
</div>

</div>
</div>

    """

    print(remove_html(doc))