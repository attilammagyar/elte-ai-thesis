import os.path
import re
import sys


def main(argv):
    if len(argv) < 3:
        print(
            f"Usage: {os.path.basename(argv[0])} thesis.tex thesis.bib",
            file=sys.stderr
        )
        return 1

    tex_filename = argv[1]
    bib_filename = argv[2]

    bib = parse_bib_file(bib_filename)
    tex_to_markdown(tex_filename, bib)

    print_bib(bib)

    return 0


def parse_bib_file(bib_filename):
    bib = {}

    with open(bib_filename, "r") as f:
        text = f.read()

    entries_split_re = re.compile(r"@(article|book|inbook|incollection|inproceedings|masterthesis|misc|phdthesis)")
    to_single_line_re = re.compile(r"\n *", flags=re.MULTILINE)

    # Not dealing with more than 1 level of nested braces.
    attribute_re = re.compile(r"[a-zA-Z0-9_-]+ *= *\{((\{[^\}]+\})*|[^{}]*)*\},?")

    id_re = re.compile(r"^ *\{ *([a-zA-Z0-9_-]+),")
    key_value_re = re.compile(r"([a-zA-Z0-9_-]+) *= *\{ *(.*) *\},?$")
    author_separator_re = re.compile(r" +and +")
    braces_re = re.compile(r"[{}]")

    for entry in entries_split_re.split(text.strip()):
        if "{" not in entry:
            continue

        entry = to_single_line_re.subn(" ", entry.strip())[0].strip()

        if entry != "":
            entry_id = id_re.search(entry)[1].strip()
            attributes = {"id": len(bib) + 1}

            for attribute in attribute_re.finditer(entry):
                attribute = attribute[0]
                parsed_attr = key_value_re.search(attribute)

                if parsed_attr:
                    key = parsed_attr[1].strip().lower()
                    value = convert_inline_formattings(
                        braces_re.sub("", parsed_attr[2].strip())
                    )

                    if key == "author":
                        authors = value.split(" and ")

                        if len(authors) > 3:
                            value = authors[0].strip() + " et. al."
                        else:
                            value = author_separator_re.sub(", ", value)

                    attributes[key] = value

            bib[entry_id] = attributes

    return bib


def convert_inline_formattings(text):
    text = text.replace("---", "&mdash;")
    text = text.replace("--", "&ndash;")
    text = re.subn(r"\\textbf\{([^}]*)\}", r"**\1**", text)[0]
    text = re.subn(r"\\emph\{([^}]*)\}", r"*\1*", text)[0]

    small_caps = []

    for text_sc in re.finditer(r"\\textsc\{([^}]*)\}", text):
        small_caps.append((text_sc[0], text_sc[1].upper()))

    for pattern, replacement in small_caps:
        text = text.replace(pattern, replacement)

    return text


def print_bib(bib):
    print("## References")
    print("")

    for citation, attributes in sorted(bib.items(), key=lambda e: e[1]["id"]):
        year = f"({attributes['year']})"
        pub = attributes.get("journal", attributes.get("booktitle", ""))

        if pub == "":
            pub = year
        else:
            pub = "*" + pub + "*"
            vol = attributes.get("volume", "")

            if vol != "":
                pub += " " + vol

            pub = pub + " " + year

        ref = ""

        for ref_type, ref_key in (("URL", "url"), ("DOI", "doi")):
            ref = (f"{ref_type}: <" + attributes.get(ref_key) + ">") if ref_key in attributes else ref

        print(f"{attributes['id']:3}. {attributes['author'].rstrip('.')}.")
        print(f"     \"{attributes['title']}\"")
        print(f"     In: {pub}.")
        print(f"     {ref}")
        print("")


def tex_to_markdown(tex_filename, bib):
    def filter_body(lines):
        found_toc = False
        found_end = False
        body = []

        for line in lines:
            if "\\tableofcontents" in line:
                found_toc = True
            elif "\\end{document}" in line:
                found_end = True
            elif (
                    found_toc
                    and not found_end
                    and "\\nocite{*}" not in line
                    and "\\newpage" not in line
                    and "\\printbibliography" not in line
            ):
                body.append(line)

        return body

    def convert_blocks(lines):
        # Note: nested lists, equations inside list items, and many other
        # things are not implemented.

        body = []

        titles = (
            (re.compile(r"\\section\{(.*)\}"), "## \\1"),
            (re.compile(r"\\subsection\{(.*)\}"), "### \\1"),
            (re.compile(r"\\subsubsection\{(.*)\}"), "#### \\1"),
            (re.compile(r"\\paragraph\{(.*)\}"), "##### \\1"),
            (re.compile(r"\\subparagraph\{(.*)\}"), "###### \\1"),
        )

        indentation_re = re.compile(r"^ *")
        eq_begin_re = re.compile(r" *(\$\$|\\begin\{(equation|align)\*?\})( *\\label{(.*)})?")
        eq_end_re = re.compile(r" *(\$\$|\\end\{(equation|align)\*?\})")

        inside_eq = False
        inside_list = False
        indentation = ""
        equations = {}
        eq_begin = None
        eq_label = None

        for line in lines:
            if inside_eq:
                if eq_end_re.search(line):
                    inside_eq = False
                    line = "\\end{align*}"

                    if eq_label is not None:
                        line = f"\\qquad\\qquad({eq_label})\n" + line
                else:
                    line = line[indentation:]
            elif inside_list:
                if "\\end{itemize}" in line:
                    inside_list = False
                    line = ""
                elif line.strip() == "":
                    line = ""
                else:
                    prefix = " * " if "\\item" in line else "   "
                    line = prefix + line.replace("\\item", "").strip()
            else:
                if "\\begin{itemize}" in line:
                    inside_list = True
                    line = ""
                elif eq_begin := eq_begin_re.search(line):
                    inside_eq = True
                    indentation = len(indentation_re.match(line)[0])
                    line = "\\begin{align*}"
                    eq_label = eq_begin[4]

                    if eq_label is not None:
                        equations[eq_label] = len(equations) + 1
                        eq_label = equations[eq_label]
                else:
                    for title_re, repl in titles:
                        line = title_re.sub(repl, line.strip())

            body.append(line)

        return body, equations

    def replace_citations(text, bib):
        for citation in set(re.findall(r"\\cite\{[^}]*\}", text, flags=re.MULTILINE)):
            md_citation = (
                "<sup>["
                + ",".join(str(bib[c]["id"]) for c in citation[6:-1].replace(" ", "").split(","))
                + "]</sup>"
            )

            text = text.replace(citation, md_citation)

        return text

    with open(tex_filename, "r") as f:
        text = f.read()

    lines = filter_body(text.split("\n"))
    lines, equations = convert_blocks(lines)

    text = "\n".join(lines)
    text = text.replace("\\newpage", "")
    text = text.replace("\\nocite{*}", "")
    text = text.replace("\\printbibliography[heading=bibintoc]", "")

    text = convert_inline_formattings(text)

    for eq_label, eq_number in equations.items():
        text = text.replace("\\ref{" + eq_label + "}", str(eq_number))

    text = replace_citations(text, bib)

    print(text.strip())
    print("")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
