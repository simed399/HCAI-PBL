# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Mohammed Boutchich", "matriculation": "21968035"},
        {"name": "Mohamed Miled", "matriculation": "201965712"},
        
    ]
    
    projects = [
        {"name": "Home", "url_name": "home:index"},
        {"name": "project1", "url_name": "project1:index"},
        {"name": "project2", "url_name": "project2:index"},
        {"name": "project3", "url_name": "project3:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))