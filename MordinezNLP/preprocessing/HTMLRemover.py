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

    text = tree.body.text(separator='\n')
    return text