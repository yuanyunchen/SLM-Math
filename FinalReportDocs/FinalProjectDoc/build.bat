@echo off
REM Build LaTeX document with bibliography
echo Building finalreport_version3.tex...

pdflatex finalreport_version3.tex
bibtex finalreport_version3
pdflatex finalreport_version3.tex
pdflatex finalreport_version3.tex

echo.
echo Build complete! Open finalreport_version3.pdf
pause

