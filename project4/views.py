from django.shortcuts import render, redirect
from .ml.mf import train_item_embeddings, infer_user_vector
from .ml.active_learning import uncertainty_query
from .utils.dataset import get_movie_list
from .models import Participant, PreSurvey, HeldOutRating, PostSurvey
import random
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Pre-compute item embeddings once
V, movie_ids = train_item_embeddings(k=20)

# Budget of ratings
BUDGET = 10

# In-memory movie metadata cache with genres
from .utils.dataset import get_movies_df
movies_df = get_movies_df()
MOVIES = {}
for _, row in movies_df.iterrows():
    MOVIES[row['movieId']] = {
        'id': row['movieId'],
        'title': row['title'],
        'genres': row['genres'].split('|') if pd.notna(row['genres']) else []
    }


def generate_explanation(u_current, movie_id):
    """
    Simple explanation: we use the movie's primary genre.
    """
    movie = MOVIES[movie_id]
    if movie['genres']:
        primary = movie['genres'][0]
        return (
            f"""
            {movie['title']} is a {primary} film. Rating it helps us understand your preference for {primary} movies,
            which will improve recommendations of similar {primary}-genre titles.
            """
        )
    else:
        return "Your rating will help personalize your future movie recommendations."
# Global variables for lazy loading
#V = None
#movie_ids = None

def _load_embeddings():
    """Load embeddings if not already loaded"""
    global V, movie_ids
    if V is None or movie_ids is None:
        try:
            V, movie_ids = train_item_embeddings(k=20)
        except ValueError:
            # Create dummy embeddings if training fails
            V = np.zeros((10, 20))  # 10 movies, 20 features
            movie_ids = np.arange(10)

def index(request):
    return render(request, 'project4/index.html')

def task1_intro(request):
    # reset session
    request.session['ratings'] = {}
    return redirect('project4:task1_next')


def task1_next(request):
    _load_embeddings()  # Ensure embeddings are loaded
    ratings = request.session.get('ratings', {})  # {movieId:rating}
    # infer current u
    if ratings:
        u = infer_user_vector(ratings)
    else:
        u = np.zeros(V.shape[1])
    # query next item
    rated_ids = [int(k) for k in ratings.keys()]  # Convert string keys to integers
    next_id = uncertainty_query(u, V, movie_ids, rated_ids, n=1)[0]
    movie_list = dict(get_movie_list())
    if next_id in movie_list:
        title = movie_list[next_id]
    else:
        # Fallback if movie ID not found
        title = f"Movie {next_id}"

    if request.method == 'POST':
        # save posted rating
        r = float(request.POST['rating'])
        ratings[str(next_id)] = r
        request.session['ratings'] = ratings
        # loop until budget (e.g. 10)
        if len(ratings) >= 10:
            return render(request, 'project4/task1_done.html', {'ratings': ratings})
        return redirect('project4:task1_next')

    # Generate explanation for why this movie was selected
    explanation = generate_explanation(u, next_id)
    
    return render(request, 'project4/task1.html', {
        'movie_id': next_id, 
        'title': title,
        'explanation': explanation,
        'step': len(ratings) + 1,
        'budget': BUDGET
    })

# STEP 1: Consent & random assignment
def study_start(request):
    if request.method == 'POST':
        arm = random.choice(['explanation', 'control'])
        p = Participant.objects.create(assigned_to=arm)
        request.session['participant_id'] = str(p.id)
        return redirect('project4:study_pre')
    return render(request, 'project4/study_start.html')

# STEP 2: Pre-survey
def study_pre_survey(request):
    pid = request.session.get('participant_id')
    p   = Participant.objects.get(id=pid)
    if request.method == 'POST':
        PreSurvey.objects.create(
            participant=p,
            age=int(request.POST['age']),
            gender=request.POST['gender'],
            familiarity=int(request.POST['familiarity'])
        )
        # initialize Task1 session
        request.session['ratings'] = {}
        request.session['show_explanation'] = (p.assigned_to=='explanation')
        return redirect('project4:study_quiz')
    return render(request, 'project4/study_pre.html')

# STEP 3: Guided Quiz (Task1) or Control
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

@method_decorator(csrf_exempt, name='dispatch')
def study_quiz(request):
    """
    1) On POST: save the posted rating into session;
       if we've hit BUDGET, go to held-out, else loop back to quiz.
    2) On GET: infer u, pick next movie, render 'study_quiz.html'.
    """
    # load or init ratings
    ratings = request.session.get('ratings', {})

    if request.method == 'POST':
        # current movie id stored in session by last GET
        mid = request.session.get('current_next_id')
        if mid:
            ratings[mid] = float(request.POST['rating'])
            request.session['ratings'] = ratings
            request.session.modified = True

        # if done, move to held-out
        if len(ratings) >= BUDGET:
            return redirect('project4:study_held')
        # otherwise loop back to GET
        return redirect('project4:study_quiz')

    # --- GET logic below ---

    # convert keys to ints
    rated_map = {int(k): v for k, v in ratings.items()}
    # infer u
    u = infer_user_vector(rated_map) if rated_map else np.zeros(V.shape[1])
    # pick next
    next_ids = uncertainty_query(u, V, movie_ids, list(rated_map.keys()), n=1)
    next_id   = int(next_ids[0])
    # store for the POST
    request.session['current_next_id'] = str(next_id)

    movie      = MOVIES[next_id]
    title      = movie['title']
    explanation = generate_explanation(u, next_id) if request.session.get('show_explanation') else ''
    step       = len(rated_map) + 1

    return render(request, 'project4/study_quiz.html', {
        'movie_id': next_id,
        'title': title,
        'explanation': explanation,
        'step': step,
        'budget': BUDGET,
        'show_explanation': bool(explanation),
    })

# STEP 4: Held-out ratings
def study_held_out(request):
    try:
        pid = request.session['participant_id']
        p   = Participant.objects.get(id=pid)
    except (KeyError, Participant.DoesNotExist):
        # Handle missing participant data
        return redirect('project4:study_start')
    
    movies = get_movie_list()
    held   = request.session.get('held_list')
    if not held:
        # sample 10 random distinct movie IDs
        held = random.sample([m[0] for m in movies], 10)
        request.session['held_list'] = held
    
    if request.method=='POST':
        # Debug: log what's in the POST data
        logger.info(f"POST data keys: {list(request.POST.keys())}")
        logger.info(f"Expected keys: {[f'r{mid}' for mid in held]}")
        
        ratings_processed = 0
        for mid in held:
            try:
                r = float(request.POST[f'r{mid}'])
                HeldOutRating.objects.create(participant=p, movie_id=mid, rating=r)
                ratings_processed += 1
            except (KeyError, ValueError) as e:
                # Handle missing or invalid rating
                logger.warning(f"Error processing rating for movie {mid}: {e}")
                continue
        
        # Only proceed if we have at least some ratings
        if ratings_processed > 0:
            return redirect('project4:study_post')
        else:
            # If no ratings were processed, stay on the same page
            logger.error("No ratings were processed successfully")
    
    # Create a list of movies with proper structure for the template
    held_movies = []
    for movie in movies:
        if movie[0] in held:
            held_movies.append({
                'id': movie[0],
                'title': movie[1]
            })
    
    return render(request, 'project4/study_held.html', {'movies': held_movies})

# STEP 5: Post-survey
def study_post_survey(request):
    pid = request.session['participant_id']
    p   = Participant.objects.get(id=pid)
    if request.method=='POST':
        PostSurvey.objects.create(
            participant=p,
            trust=int(request.POST['trust']),
            transparency=int(request.POST['transparency']),
            satisfaction=int(request.POST['satisfaction'])
        )
        return redirect('project4:study_thanks')
    return render(request, 'project4/study_post.html')

# STEP 6: Thank-you page
def study_thanks(request):
    return render(request, 'project4/study_thanks.html')