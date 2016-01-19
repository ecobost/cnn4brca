READ ME FILE
============

This file aims at providing general (and hopefully sufficient) information regarding the phdDissertationFormat.sty file, which is to be used when typesetting your doctoral dissertation for the Information Technologies and Communications Graduate Program.

Folder Contents
---------------

The contents that you will find in this folder are:

- phdThesisFormat.sty (The style file to be used when preparing your dissertation. Important file!).
- DTC-PhD-Thesis.tex (This is the main LaTeX file where all the settings are set.  This is the file you modify with your own information).
- Chapter 01.tex and Chapter 02.tex (These are sample files of how to start and type in a chapter of your dissertation).
- abstractFile.tex (The file where your abstract is to be placed).
- acknowledgementFile.tex (This is the file where you acknowledge all those who deserve so).
- dedicationFile.tex (The file that informs the reader whom you dedicate this dissertation to).
- vitaFile.tex (Here you provide a brief description of your CV).
- Appendix 01.tex (An example of how to prepare an appendix).
- escudo-itesm.png (This file is needed in order to generate the dissertation using pdflatex.  Not to be erased!).
- escudo-itesm.eps (This file is needed in order to generate the dissertation using latex.  Not to be erased!).
- References.bib (A sample file that provides some examples of bibtex entries)
- exampleMainFile.pdf (The outcome of executing pdflatex to the main tex file using the sty file)

Obtaining you pdf dissertation document
----------------------------------------

The style file and the exampleMainFile.tex one assume you are using "pdflatex" to generate your documents.  I recommend that your use this tool to obtain your pdf final document.  One important reason is the use of real True Type Fonts by default.  Regarding figures, you do need pdf figures, png, jpg.  Almost all of the figures that are going to be used in the dissertation are either jpg or png, and pdflatex works with those just perfectly.  How to obtain a pdf figure?  The figures that may need some work are those that represent results or observations or our analysis (e.g., an eps generated file using gnuplot).  Under Linux, it is easy to convert an eps file to pdf using the eps2pdf tool in the console.  Just type in:

\> epstopdf input.eps output.pdf

and you will get your figure in pdf format.

If you are working in Windows, you can change your files to almost any format using Gimp (this is also true if you are working under Linux).  Also, Adobe Acrobat Professional can do the trick (But this costs money, we want to avoid this).  There are also free ways to get a pdf figure or document and the instructions are in this site:

http://kenchiro.tripod.com/howtoPDF.html

Just follow the instructions and you will be set and this is all about figures.

Another reason why I recommend the use of pdflatex is the ability that this tool gives us to set up the size of our output.  The style file has already set the size, as long as pdflatex is used.  If you go from dvi to ps to pdf there are some extra steps to follow in order to ensure that the size of your dissertation document is correct and appropriate.

Under Linux do this:

\> latex sample
\> dvips -Ppdf -G0 -tletter sample
\> ps2pdf -dCompatibilityLevel=1.4 -dMAxSubsetPct=100 -dSubsetFonts=true -dEmbedAllFonts=true -sPAPERSIZE=letter sample.ps

Under Windows follow the next steps (You need to have Ghostscript with GSview):

The first step is to set the media size to Letter. To do this, in the GSview menubar at the top click on "Media" and select Letter.

Next, click on "File | Convert" and open the ps file you want to convert to pdf. In the ensuing window, in Options textbox type in "-dMAxSubsetPct=100," and in the same window make sure the pdfwrite resolution is set to the highest value.

Then, on the same window click on the "Properties" button.  Under the Property window, choose "CompatibilityLevel" and choose the highest one.  After, under the same property window, choose "EmbedAllFonts" and set its value to "true".  Finally, also in the property window, choose the "SubsetFonts" and set its value to "true."  Click OK and you will be asked for the location where your output file is to be placed.

If you are a Windows user, TeXnicCenter is an IDE that helps you a lot when preparing latex documents.  If, on the contrary, you are a Linux user, Kile is the way to go.  Moreover, both IDEs offer you the possibility to use pdflatex.

Doubts
------

When in doubt, do not hesitate to contact me: Luis Marcelo Fernández Carrasco.  I guess you already know where to find me or send an email to luis_marcelo_f <at> hotmail <dot> com.
