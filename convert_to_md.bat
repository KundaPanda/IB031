call activate ml
jupyter nbconvert --to html --output "index" notebook.ipynb
jupyter nbconvert --to markdown --output "index" notebook.ipynb
jupyter nbconvert --to pdf notebook.ipynb
pause
