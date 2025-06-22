from django.shortcuts import render

def index(request):
    
    return render(request, "project3/index.html", {})
