import io
import langdetect
from bs4 import BeautifulSoup
from html2text import HTML2Text
from html2text import html2text
import urllib.request


def getContent(toCheck: str, file: bool) -> str:
    """ Given an HTML file or website URL, retrieves the content.
    Setting parameter \"file\" to true will make the function handle \"toCheck\" as a filename. Otherwise, it is a URL.
    Depends on html2text and urllib.request library."""
    htmlHandler = HTML2Text()
    htmlHandler.ignore_links = True
    text = ""
    html = ""
    if file:
        html = getFileHTML(toCheck)
    else:
        html = getHTML(toCheck)
    text = htmlHandler.handle(html)
    return text


def getMetacontent(toCheck: str, file: bool) -> dict:
    """ Given an HTML file or website URL, retrieves the title, meta tags, external references, ID names, and class names.
    Setting parameter \"file\" to true will make the function handle \"toCheck\" as a filename. Otherwise, it is a URL.
    Depends on urllib.request library."""
    htmlSoup = ""
    if file:
        # use BeautifulSoup to get a DOM of the HTMl
        # try unicode encoding first
        try:
            with io.open(toCheck, encoding="utf8") as file:
                htmlSoup = BeautifulSoup(file, 'html.parser')
        # otherwise, try latin (portuegese and spanish)
        except UnicodeDecodeError:
            with io.open(toCheck, encoding="latin-1") as file:
                htmlSoup = BeautifulSoup(file, 'html.parser')
    else:
        html = getHTML(toCheck)
        htmlSoup = BeautifulSoup(html, 'html.parser')

    title = ""
    if htmlSoup.title is not None:
        title = htmlSoup.title.string

    # Handle meta tags
    meta = []
    metaElements = htmlSoup.findAll("meta")
    for tag in metaElements:
        meta.append(list(tag.attrs.values()))

    # Handle external references
    refs = []
    refElements = htmlSoup.find_all(href=True)
    for element in refElements:
        ref = element.get('href')
        refs.append(ref)
    # Handle source references
    refElements = htmlSoup.find_all(src=True)
    for element in refElements:
        ref = element.get('src')
        refs.append(ref)

    # Handle 'className'
    classNames = []
    elementsWithClassName = htmlSoup.find_all(class_=True)
    for element in elementsWithClassName:
        className = element.get('class')
        classNames.extend(className)
    # Remove duplicates
    classNames = list(set(classNames))

    # Handle 'idName'
    idNames = []
    for tag in htmlSoup.findAll(True, {'id': True}):
        idNames.append(tag['id'])
    # Remove duplicates
    idNames = list(set(idNames))

    content = {"meta": meta, "title": title, "externalReferences": refs, "idNames": idNames, "classNames": classNames}
    return content


def getFileHTML(filename: str) -> str:
    """Retrieves the HTML of a given file.
    Format to decode is UTF-8 by default, but can be changed
    with decodeFormat kwarg."""
    # In case UTF-8 encoding fails to read the file, try Latin-1 encoding
    try:
        with io.open(filename, encoding="utf8") as file:
            html = file.read()
    except UnicodeDecodeError:
        with io.open(filename, encoding="latin-1") as file:
            html = file.read()
    return html


def getHTML(url: str, decodeFormat="utf8") -> str:
    """Retrieves the HTML of a given URL.
    Format to decode is UTF-8 by default, but can be changed
    with decodeFormat kwarg."""
    # https://stackoverflow.com/questions/24153519/how-to-read-html-from-a-url-in-python-3
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    # when decoding, skip any bytes that do not match our decoding format
    mystr = mybytes.decode(decodeFormat, 'replace')
    fp.close()
    return mystr


def isEnglish(content: str) -> bool:
    """Tentatively: uses langdetect to determine if content is primarily in English
    or not. """
    return True if langdetect.detect(content) == 'en' else False


def debug() -> None:
    print(getMetacontent("randomHTML.txt"))
