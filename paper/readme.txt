for editing a tex file

detex -n /tmp/sammba_detex/sammba_frontiers.tex > /tmp/sammba_frontiers_detexed.txt

for constituting the pdf with minted and Times New Roman font
xelatex -shell-escape sammba_frontiers.tex


for counting the number of words
xelatex -shell-escape sammba_frontiers.tex
