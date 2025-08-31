from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .ml.mf import train_item_embeddings, infer_user_vector, predict_ratings, get_recommendations
from .ml.active_learning import uncertainty_query
from .utils.dataset import get_movie_list, get_movies_df
from .models import Participant, PreSurvey, HeldOutRating, PostSurvey, QuizRating, StudySession, Feedback
import random
import logging
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from django.utils import timezone
import re


logger = logging.getLogger(__name__)

# Pre-compute item embeddings once
V, movie_ids = train_item_embeddings(k=20)

# Budget of ratings for each phase
STANDARD_BUDGET = 10  # First phase: standard study
GUIDE_BUDGET = 10     # Second phase: guided study
TOTAL_BUDGET = STANDARD_BUDGET + GUIDE_BUDGET

# In-memory movie metadata cache with genres
from .utils.dataset import get_movies_df
movies_df = get_movies_df()
MOVIES = {}

def extract_year_from_title(title):
    """Extract year from movie title like 'Movie Name (1999)'"""
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else 2000

for _, row in movies_df.iterrows():
    MOVIES[row['movieId']] = {
        'id': row['movieId'],
        'title': row['title'],
        'genres': row['genres'].split('|') if pd.notna(row['genres']) else [],
        'year': extract_year_from_title(row['title'])
    }


def generate_explanation(u_current, movie_id):
    """
    Enhanced explanation: use the movie's primary genre and explain impact.
    """
    movie = MOVIES[movie_id]
    if movie['genres']:
        primary = movie['genres'][0]
        return (
            f"{movie['title']} is a {primary} film. Rating it will help us understand "
            f"your preference for {primary} movies and improve recommendations of similar "
            f"{primary}-genre titles. Your rating will directly influence which movies "
            f"appear in your personalized recommendations."
        )
    else:
        return ("Your rating will help personalize your future movie recommendations by "
                "learning your taste preferences.")

def generate_recommendations(user_vector, n=10):
    """Generate top N recommendations with explanations"""
    # Get recommendations using ML function
    recommended_movie_ids, recommended_scores = get_recommendations(
        user_vector, V, movie_ids, n=n
    )
    
    recommendations = []
    for movie_id, score in zip(recommended_movie_ids, recommended_scores):
        if movie_id in MOVIES:
            movie = MOVIES[movie_id]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'pred': score,
                'genres': ', '.join(movie['genres'][:2]) if movie['genres'] else 'Unknown',
                'year': movie['year'],
                'because': "movies you rated highly",
                'tags': ', '.join(movie['genres'][:2]) if movie['genres'] else 'Unknown'
            })
    
    return recommendations

def calculate_rmse(participant):
    """Calculate RMSE on held-out ratings"""
    try:
        session = StudySession.objects.get(participant=participant)
        user_vector = np.array(json.loads(session.user_vector))
        
        held_out_ratings = HeldOutRating.objects.filter(participant=participant)
        if not held_out_ratings.exists():
            return None
            
        actual_ratings = []
        predicted_ratings = []
        
        for rating in held_out_ratings:
            if rating.movie_id in movie_ids:
                movie_idx = np.where(movie_ids == rating.movie_id)[0]
                if len(movie_idx) > 0:
                    pred = np.dot(user_vector, V[movie_idx[0]])
                    actual_ratings.append(rating.rating)
                    predicted_ratings.append(pred)
        
        if len(actual_ratings) > 0:
            rmse = np.sqrt(np.mean((np.array(actual_ratings) - np.array(predicted_ratings))**2))
            return rmse
    except Exception as e:
        logger.error(f"Error calculating RMSE: {e}")
    return None

def analyze_user_preferences(user_vector, quiz_ratings_dict):
    """Analyze user preferences from ratings and vector"""
    if not quiz_ratings_dict:
        return {'likes': 'Sci‑Fi, Mind‑benders, Thrillers', 'dislikes': 'Slapstick comedy, Gore'}
    
    # Get genre preferences based on ratings
    genre_ratings = {}
    for movie_id, rating in quiz_ratings_dict.items():
        if movie_id in MOVIES:
            movie = MOVIES[movie_id]
            for genre in movie['genres']:
                if genre not in genre_ratings:
                    genre_ratings[genre] = []
                genre_ratings[genre].append(rating)
    
    # Calculate average ratings per genre
    genre_averages = {}
    for genre, ratings in genre_ratings.items():
        genre_averages[genre] = np.mean(ratings)
    
    # Sort genres by preference
    sorted_genres = sorted(genre_averages.items(), key=lambda x: x[1], reverse=True)
    
    # Extract likes (top 3) and dislikes (bottom 2)
    likes = [genre for genre, avg in sorted_genres[:3] if avg >= 3.5]
    dislikes = [genre for genre, avg in sorted_genres[-2:] if avg < 3.0]
    
    return {
        'likes': ', '.join(likes) if likes else 'Adventure, Drama',
        'dislikes': ', '.join(dislikes) if dislikes else 'Horror, Documentary'
    }

def get_next_movie_title():
    """Get a random movie title for the 'what if' section"""
    sample_movies = ['The Matrix (1999)', 'Inception (2010)', 'Pulp Fiction (1994)', 'The Godfather (1972)']
    return random.choice(sample_movies)

def analyze_genre_preferences(user_vector, top_genres=8):
    """
    Analyze which movie genres the user is likely to be recommended based on their user vector.
    Returns genre preference scores for spider chart visualization.
    """
    if user_vector is None or len(user_vector) == 0:
        print("DEBUG: User vector is None or empty")
        return {}
    
    print(f"DEBUG: Starting genre analysis with user vector: {user_vector[:3] if len(user_vector) > 0 else 'empty'}")
    
    # Define main movie genres for analysis
    main_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    
    genre_scores = {}
    raw_scores = {}  # Store raw predicted ratings for proper normalization
    
    # Ensure we have the required global variables
    if V is None or movie_ids is None:
        print("DEBUG: V or movie_ids is None")
        return {genre: 50.0 for genre in main_genres}
    
    print(f"DEBUG: V shape: {V.shape}, movie_ids length: {len(movie_ids)}, MOVIES length: {len(MOVIES)}")
    
    # First pass: calculate raw average predicted ratings for each genre
    for genre in main_genres:
        genre_movie_ids = []
        
        # Find movies with this genre
        for movie_id in movie_ids:
            if movie_id in MOVIES:
                movie = MOVIES[movie_id]
                if movie.get('genres') and genre in movie['genres']:
                    genre_movie_ids.append(movie_id)
        
        print(f"DEBUG: Genre {genre} has {len(genre_movie_ids)} movies")
        
        if genre_movie_ids:
            # Get movie indices for this genre
            genre_indices = []
            for mid in genre_movie_ids:
                movie_idx = np.where(movie_ids == mid)[0]
                if len(movie_idx) > 0:
                    genre_indices.append(movie_idx[0])
            
            if genre_indices:
                # Calculate predicted ratings for movies in this genre
                genre_V = V[genre_indices]
                predicted_ratings = np.dot(genre_V, user_vector)
                
                # Use average predicted rating as genre preference score
                avg_rating = np.mean(predicted_ratings)
                raw_scores[genre] = avg_rating
                
                print(f"DEBUG: Genre {genre} - {len(genre_indices)} valid movies, avg predicted rating: {avg_rating}")
                print(f"DEBUG: Genre {genre} - predicted ratings range: {np.min(predicted_ratings):.3f} to {np.max(predicted_ratings):.3f}")
            else:
                # No valid movie indices found for this genre
                print(f"DEBUG: Genre {genre} - no valid indices, using neutral score")
        else:
            # No movies found for this genre
            print(f"DEBUG: Genre {genre} - no movies found, using neutral score")
    
    # Second pass: normalize scores to 0-100 scale based on relative differences
    if raw_scores:
        min_score = min(raw_scores.values())
        max_score = max(raw_scores.values())
        score_range = max_score - min_score
        
        print(f"DEBUG: Raw score range: {min_score:.3f} to {max_score:.3f}, range: {score_range:.3f}")
        
        # Use a wider spread to better show differences
        # Map the lowest score to 0% and highest to 100%, unless range is too small
        min_display = 0.0
        max_display = 100.0
        
        for genre in main_genres:
            if genre in raw_scores:
                if score_range > 0.01:  # Only normalize if there's meaningful difference
                    # Linear normalization to 0-100 scale that preserves relative differences
                    normalized = ((raw_scores[genre] - min_score) / score_range) * (max_display - min_display) + min_display
                    # Round to 1 decimal
                    normalized_score = round(max(0.0, min(100.0, normalized)), 1)
                else:
                    # If all scores are very similar, use 50% for all
                    normalized_score = 50.0
                
                genre_scores[genre] = normalized_score
                print(f"DEBUG: Genre {genre}: raw={raw_scores[genre]:.3f} -> normalized={normalized_score:.1f}%")
            else:
                # No data for this genre, use middle value
                genre_scores[genre] = 50.0
                print(f"DEBUG: Genre {genre} - no data, using neutral score")
    else:
        # No genre data found, return neutral scores
        genre_scores = {genre: 50.0 for genre in main_genres}
        print("DEBUG: No genre data found, using neutral scores for all")
    
    print(f"DEBUG: Final genre scores: {genre_scores}")
    return genre_scores

def analyze_genre_preferences_direct_ratings(all_ratings_dict):
    """
    Alternative approach: analyze genre preferences directly from user ratings
    without using the matrix factorization user vector. This should more directly
    reflect the actual ratings given by the user.
    """
    if not all_ratings_dict:
        print("DEBUG: No ratings provided for direct analysis")
        return {}
    
    print(f"DEBUG: Direct analysis with ratings: {all_ratings_dict}")
    
    # Define main movie genres for analysis
    main_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    
    # Calculate average rating per genre based on actual user ratings
    genre_ratings = {genre: [] for genre in main_genres}
    
    for movie_id, rating in all_ratings_dict.items():
        if movie_id in MOVIES:
            movie = MOVIES[movie_id]
            for genre in movie.get('genres', []):
                if genre in genre_ratings:
                    genre_ratings[genre].append(rating)
    
    # Calculate average ratings and normalize
    genre_averages = {}
    for genre, ratings in genre_ratings.items():
        if ratings:
            genre_averages[genre] = np.mean(ratings)
            print(f"DEBUG: Genre {genre}: {len(ratings)} ratings, avg={genre_averages[genre]:.2f}")
    
    # Normalize to 0-100 scale based on rating range (0.5-5.0)
    genre_scores = {}
    if genre_averages:
        min_possible_rating = 0.5
        max_possible_rating = 5.0
        rating_range = max_possible_rating - min_possible_rating
        
        for genre in main_genres:
            if genre in genre_averages:
                # Normalize based on the full rating scale (0.5-5.0 -> 0-100)
                normalized = ((genre_averages[genre] - min_possible_rating) / rating_range) * 100
                genre_scores[genre] = round(max(0.0, min(100.0, normalized)), 1)
                print(f"DEBUG: Genre {genre}: avg_rating={genre_averages[genre]:.2f} -> normalized={genre_scores[genre]:.1f}%")
            else:
                # No ratings for this genre, use neutral score
                genre_scores[genre] = 50.0
                print(f"DEBUG: Genre {genre}: no ratings, using neutral score")
    else:
        # No genre averages calculated, return neutral scores
        genre_scores = {genre: 50.0 for genre in main_genres}
        print("DEBUG: No genre averages calculated, using neutral scores")
    
    print(f"DEBUG: Final direct genre scores: {genre_scores}")
    return genre_scores
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
    """Enhanced landing page"""
    return render(request, 'project4/index.html', {
        'has_pdf': True,
        'study_available': True
    })

def download_study_guide(request):
    """Download the User Study Guide PDF"""
    try:
        # Get the path to the PDF file in the project4 directory
        pdf_path = os.path.join(os.path.dirname(__file__), 'User Study Guide.pdf')
        
        if os.path.exists(pdf_path):
            response = FileResponse(
                open(pdf_path, 'rb'),
                as_attachment=True,
                filename='User Study Guide.pdf'
            )
            response['Content-Type'] = 'application/pdf'
            return response
        else:
            return HttpResponse('PDF file not found.', status=404)
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        return HttpResponse('Error downloading file.', status=500)

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
        'budget': 10
    })

# STEP 1: Consent & random assignment
def study_start(request):
    if request.method == 'POST':
        arm = random.choice(['explanation', 'control'])
        display_name = request.POST.get('display_name', f"User_{random.randint(1000, 9999)}")
        p = Participant.objects.create(assigned_to=arm, display_name=display_name)
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

def study_quiz(request):
    """
    STANDARD STUDY phase - First 10 movies without explanations
    1) On POST: save the posted rating into session and QuizRating model;
       if we've hit STANDARD_BUDGET, go to guided study, else loop back to quiz.
    2) On GET: infer u, pick next movie, render 'study_standard.html'.
    """
    pid = request.session.get('participant_id')
    if not pid:
        return redirect('project4:study_start')
    
    participant = Participant.objects.get(id=pid)
    
    # load or init ratings
    ratings = request.session.get('ratings', {})

    if request.method == 'POST':
        # current movie id stored in session by last GET
        mid = request.session.get('current_next_id')
        if mid:
            rating_value = float(request.POST['rating'])
            ratings[mid] = rating_value
            request.session['ratings'] = ratings
            request.session.modified = True
            
            # Save to QuizRating model - mark as standard study phase
            QuizRating.objects.create(
                participant=participant,
                movie_id=int(mid),
                rating=rating_value,
                iteration=len(ratings),
                explanation_shown=False  # Standard study has no explanations
            )

        # if done with standard phase, generate recommendations and show them
        if len(ratings) >= STANDARD_BUDGET:
            # Generate recommendations based on standard study ratings
            rated_map = {int(k): v for k, v in ratings.items()}
            user_vector = infer_user_vector(rated_map)
            
            # Get 10 movie recommendations for the standard study evaluation
            recommended_movie_ids, recommended_scores = get_recommendations(
                user_vector, V, movie_ids, n=GUIDE_BUDGET, exclude_ids=list(rated_map.keys())
            )
            
            # Store recommendations for standard phase evaluation
            request.session['standard_recommendations'] = {
                'movie_ids': recommended_movie_ids.tolist() if hasattr(recommended_movie_ids, 'tolist') else list(recommended_movie_ids),
                'scores': recommended_scores.tolist() if hasattr(recommended_scores, 'tolist') else list(recommended_scores)
            }
            
            return redirect('project4:standard_interest')
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
    step       = len(rated_map) + 1

    return render(request, 'project4/study_standard.html', {
        'movie_id': next_id,
        'title': title,
        'step': step,
        'budget': STANDARD_BUDGET,
        'phase': 'standard',
    })

# Standard study interest rating - rate recommended movies
def standard_interest(request):
    """
    Standard study interest rating - user rates how interested they are in watching
    the 10 recommended movies (all on one page)
    """
    pid = request.session.get('participant_id')
    if not pid:
        return redirect('project4:study_start')
    
    participant = Participant.objects.get(id=pid)
    
    # Get recommendations from session
    standard_recommendations = request.session.get('standard_recommendations')
    if not standard_recommendations:
        return redirect('project4:study_start')
    
    # Prepare movie details for display
    recommendations = []
    for i, movie_id in enumerate(standard_recommendations['movie_ids']):
        if movie_id in MOVIES:
            movie = MOVIES[movie_id]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'score': standard_recommendations['scores'][i],
                'genres': ', '.join(movie['genres'][:2]) if movie['genres'] else 'Unknown',
                'year': movie['year']
            })
    
    if request.method == 'POST':
        # Process interest ratings
        standard_interest_ratings = {}
        for rec in recommendations:
            movie_id = rec['movie_id']
            try:
                interest_rating = float(request.POST[f'interest_{movie_id}'])
                standard_interest_ratings[str(movie_id)] = interest_rating
                
                # Save to database (using HeldOutRating model for now, can create new model later)
                HeldOutRating.objects.create(
                    participant=participant, 
                    movie_id=movie_id, 
                    rating=interest_rating
                )
            except (KeyError, ValueError):
                continue
        
        # Store interest ratings in session
        request.session['standard_interest_ratings'] = standard_interest_ratings
        
        # Move to guided study phase
        return redirect('project4:study_guided_start')
    
    return render(request, 'project4/standard_interest.html', {
        'recommendations': recommendations,
        'phase': 'standard',
        'phase_name': 'Standard Study'
    })


# Guided study start - transition between phases
def study_guided_start(request):
    """
    Transition page to start guided study phase
    """
    pid = request.session.get('participant_id')
    if not pid:
        return redirect('project4:study_start')
    
    if request.method == 'POST':
        # Initialize guided study phase
        request.session['guided_ratings'] = {}
        return redirect('project4:study_guided')
    
    return render(request, 'project4/guided_start.html', {
        'standard_complete': True
    })

# Guided study phase - rate movies with explanations
@method_decorator(csrf_exempt, name='dispatch')
def study_guided(request):
    """
    GUIDED STUDY phase - Rate 10 movies with explanations about how it affects recommendations
    """
    pid = request.session.get('participant_id')
    if not pid:
        return redirect('project4:study_start')
    
    participant = Participant.objects.get(id=pid)
    
    # Get guided ratings
    guided_ratings = request.session.get('guided_ratings', {})
    
    if request.method == 'POST':
        # current movie id stored in session by last GET
        mid = request.session.get('current_guided_id')
        if mid:
            rating_value = float(request.POST['rating'])
            guided_ratings[mid] = rating_value
            request.session['guided_ratings'] = guided_ratings
            request.session.modified = True
            
            # Save to QuizRating model - mark as guided study phase
            QuizRating.objects.create(
                participant=participant,
                movie_id=int(mid),
                rating=rating_value,
                iteration=STANDARD_BUDGET + len(guided_ratings),  # Continue numbering after standard
                explanation_shown=True  # Guided study has explanations
            )

        # if done with guided phase, generate recommendations for guided study
        if len(guided_ratings) >= GUIDE_BUDGET:
            # Generate recommendations based on guided study ratings
            rated_map = {int(k): v for k, v in guided_ratings.items()}
            user_vector = infer_user_vector(rated_map)
            
            # Get 10 movie recommendations for the guided study evaluation
            # Exclude both standard and guided rated movies
            standard_ratings = request.session.get('ratings', {})
            all_excluded = list(rated_map.keys()) + [int(k) for k in standard_ratings.keys()]
            
            recommended_movie_ids, recommended_scores = get_recommendations(
                user_vector, V, movie_ids, n=GUIDE_BUDGET, exclude_ids=all_excluded
            )
            
            # Store recommendations for guided phase evaluation
            request.session['guided_recommendations'] = {
                'movie_ids': recommended_movie_ids.tolist() if hasattr(recommended_movie_ids, 'tolist') else list(recommended_movie_ids),
                'scores': recommended_scores.tolist() if hasattr(recommended_scores, 'tolist') else list(recommended_scores)
            }
            
            return redirect('project4:guided_interest')
        # otherwise loop back to GET
        return redirect('project4:study_guided')

    # --- GET logic below ---
    
    # Use uncertainty sampling to pick next movie (like standard study but with explanations)
    rated_map = {int(k): v for k, v in guided_ratings.items()}
    u = infer_user_vector(rated_map) if rated_map else np.zeros(V.shape[1])
    
    # Exclude movies from standard study to avoid duplicates
    standard_ratings = request.session.get('ratings', {})
    excluded_ids = list(rated_map.keys()) + [int(k) for k in standard_ratings.keys()]
    
    next_ids = uncertainty_query(u, V, movie_ids, excluded_ids, n=1)
    next_id = int(next_ids[0])
    
    # Store for the POST
    request.session['current_guided_id'] = str(next_id)

    movie = MOVIES[next_id]
    title = movie['title']
    
    # Generate explanation for guided study
    explanation = generate_explanation(u, next_id)
    
    # Calculate current step before using it
    step = len(guided_ratings) + 1
    
    # Calculate genre preferences for spider chart
    # For step 1, show empty chart with message
    if step == 1:
        genre_preferences = {}
        show_spider_chart = False
        spider_message = "Your preference spider chart will appear after your first rating!"
    else:
        # For spider chart, combine all ratings from standard + guided studies for complete user profile
        all_ratings = {}
        all_ratings.update({int(k): v for k, v in standard_ratings.items()})
        all_ratings.update(rated_map)
        
        # Use combined user vector for more accurate genre preferences
        combined_u = infer_user_vector(all_ratings) if all_ratings else np.zeros(V.shape[1])
        
        # Debug: Log the actual ratings and user vector
        print(f"DEBUG: Combined ratings: {all_ratings}")
        print(f"DEBUG: User vector shape: {combined_u.shape if hasattr(combined_u, 'shape') else 'no shape'}")
        print(f"DEBUG: User vector sample: {combined_u[:5] if len(combined_u) > 0 else 'empty'}")
        
        # Use direct ratings analysis for more accurate genre preferences
        # This approach directly uses the user's actual ratings instead of matrix factorization predictions
        genre_preferences = analyze_genre_preferences_direct_ratings(all_ratings)
        print(f"DEBUG: Direct genre preferences result: {genre_preferences}")
        
        # Fallback to matrix factorization approach if direct analysis fails
        if not genre_preferences or all(score == 50.0 for score in genre_preferences.values()):
            print("DEBUG: Direct analysis failed or returned neutral scores, trying matrix factorization approach")
            genre_preferences = analyze_genre_preferences(combined_u)
            print(f"DEBUG: Matrix factorization genre preferences result: {genre_preferences}")
        
        # Always show the chart with real data - no fallback
        show_spider_chart = True
        spider_message = "Based on all your ratings so far, here's what genres you're likely to be recommended:"

    return render(request, 'project4/study_guided.html', {
        'movie_id': next_id,
        'title': title,
        'explanation': explanation,
        'step': step,
        'budget': GUIDE_BUDGET,
        'phase': 'guided',
        'total_step': STANDARD_BUDGET + step,
        'total_budget': TOTAL_BUDGET,
        'genre_preferences': genre_preferences,
        'show_spider_chart': show_spider_chart,
        'spider_message': spider_message,
        'genre_preferences_json': json.dumps(genre_preferences),
    })

# Guided study interest rating - rate recommended movies
def guided_interest(request):
    """
    Guided study interest rating - user rates how interested they are in watching
    the 10 recommended movies (all on one page)
    """
    pid = request.session.get('participant_id')
    if not pid:
        return redirect('project4:study_start')
    
    participant = Participant.objects.get(id=pid)
    
    # Get recommendations from session
    guided_recommendations = request.session.get('guided_recommendations')
    if not guided_recommendations:
        return redirect('project4:study_start')
    
    # Prepare movie details for display
    recommendations = []
    for i, movie_id in enumerate(guided_recommendations['movie_ids']):
        if movie_id in MOVIES:
            movie = MOVIES[movie_id]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'score': guided_recommendations['scores'][i],
                'genres': ', '.join(movie['genres'][:2]) if movie['genres'] else 'Unknown',
                'year': movie['year']
            })
    
    if request.method == 'POST':
        # Process interest ratings
        guided_interest_ratings = {}
        for rec in recommendations:
            movie_id = rec['movie_id']
            try:
                interest_rating = float(request.POST[f'interest_{movie_id}'])
                guided_interest_ratings[str(movie_id)] = interest_rating
            except (KeyError, ValueError):
                continue
        
        # Store interest ratings in session
        request.session['guided_interest_ratings'] = guided_interest_ratings
        
        # Move to results
        return redirect('project4:study_results')
    
    return render(request, 'project4/guided_interest.html', {
        'recommendations': recommendations,
        'phase': 'guided',
        'phase_name': 'Guided Study'
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
        return redirect('project4:study_results')  # Changed to go to results page
    return render(request, 'project4/study_post.html')

# Legacy thank you page (now redirects to results)
def study_thanks(request):
    return redirect('project4:study_results')

# STEP 6: Enhanced Thank-you page with comprehensive results
def study_results(request):
    """Comprehensive results page based on provided template"""
    try:
        pid = request.session.get('participant_id')
        if not pid:
            return redirect('project4:study_start')
            
        participant = Participant.objects.get(id=pid)
        
        # Get all ratings from both quiz phases
        quiz_ratings = QuizRating.objects.filter(participant=participant).order_by('iteration')
        held_out_ratings = HeldOutRating.objects.filter(participant=participant)
        
        # Get or create study session
        session, created = StudySession.objects.get_or_create(participant=participant)
        
        # Calculate user vector from all quiz ratings (standard + guided)
        standard_ratings = request.session.get('ratings', {})
        guided_ratings = request.session.get('guided_ratings', {})
        
        # Get interest ratings from both phases for comparison
        standard_interest = request.session.get('standard_interest_ratings', {})
        guided_interest = request.session.get('guided_interest_ratings', {})
        
        # Calculate average interest ratings for comparison
        standard_interest_avg = np.mean([float(v) for v in standard_interest.values()]) if standard_interest else 0
        guided_interest_avg = np.mean([float(v) for v in guided_interest.values()]) if guided_interest else 0
        
        # Calculate total interest points
        standard_interest_total = sum([float(v) for v in standard_interest.values()]) if standard_interest else 0
        guided_interest_total = sum([float(v) for v in guided_interest.values()]) if guided_interest else 0
        
        # Determine which phase produced better recommendations
        better_phase = "Guided" if guided_interest_avg > standard_interest_avg else "Standard"
        interest_improvement = guided_interest_avg - standard_interest_avg
        # Separate analysis for standard vs guided phases
        standard_ratings_dict = {int(k): v for k, v in standard_ratings.items()}
        guided_ratings_dict = {int(k): v for k, v in guided_ratings.items()}
        
        # Calculate average ratings for comparison
        standard_avg = np.mean(list(standard_ratings_dict.values())) if standard_ratings_dict else 0
        guided_avg = np.mean(list(guided_ratings_dict.values())) if guided_ratings_dict else 0
        
        # Get recommended movies information
        recommended_movies = request.session.get('recommended_movies', {})
        
        # Calculate prediction accuracy (how close our predictions were)
        prediction_accuracy = None
        if recommended_movies and guided_ratings_dict:
            predicted_scores = recommended_movies.get('scores', [])
            actual_ratings = []
            predicted_ratings = []
            
            for i, movie_id in enumerate(recommended_movies.get('movie_ids', [])):
                if movie_id in guided_ratings_dict and i < len(predicted_scores):
                    actual_ratings.append(guided_ratings_dict[movie_id])
                    predicted_ratings.append(predicted_scores[i])
            
            if actual_ratings and predicted_ratings:
                # Calculate MAE (Mean Absolute Error)
                mae = np.mean(np.abs(np.array(actual_ratings) - np.array(predicted_ratings)))
                prediction_accuracy = f"{mae:.3f}"
        
        # Combine all quiz ratings for user vector calculation
        all_quiz_ratings = {**standard_ratings_dict, **guided_ratings_dict}
        
        if all_quiz_ratings:
            user_vector = infer_user_vector(all_quiz_ratings)
            session.user_vector = json.dumps(user_vector.tolist())
            session.completion_status = 'completed'
            session.completed_at = timezone.now()
            session.save()
        else:
            user_vector = np.zeros(V.shape[1])
        
        # Generate recommendations
        recommendations = generate_recommendations(user_vector, n=8)
        
        # Calculate RMSE
        rmse = calculate_rmse(participant)
        if rmse:
            session.final_rmse = rmse
            session.save()
        
        # Analyze learned preferences separately for each phase
        standard_preferences = analyze_user_preferences(user_vector, standard_ratings_dict) if standard_ratings_dict else {'likes': 'No data', 'dislikes': 'No data'}
        guided_preferences = analyze_user_preferences(user_vector, guided_ratings_dict) if guided_ratings_dict else {'likes': 'No data', 'dislikes': 'No data'}
        
        # Also analyze combined preferences
        combined_preferences = analyze_user_preferences(user_vector, all_quiz_ratings)
        
        # Calculate progress
        progress_pct = 100  # Study is complete
        
        context = {
            'user_display_name': participant.display_name,
            'participant': participant,
            'n_ratings': len(all_quiz_ratings),
            'target_n_ratings': TOTAL_BUDGET,
            'strategy_name': "Standard vs Guided Comparison",
            'batch_size': 1,
            'k_latent': V.shape[1],
            'lambda_reg': 0.05,
            'rmse': f"{rmse:.3f}" if rmse else "—",
            'baseline_rmse': "1.2",  # Typical baseline
            'recommendations': recommendations,
            'next_movie_title': get_next_movie_title(),
            'progress_pct': progress_pct,
            'steps_done': 5,
            'steps_total': 5,
            'n_loops': len(all_quiz_ratings),
            'learned_preferences': combined_preferences,
            'standard_preferences': standard_preferences,
            'guided_preferences': guided_preferences,
            'session_id': str(participant.id)[:8],
            'u_vector': f"[{', '.join([f'{x:.2f}' for x in user_vector[:3]])}...]",
            'seed': 42,
            'year': datetime.now().year,
            'pdf_download_url': '#',  # TODO: Add actual PDF generation
            'export_csv_url': '#',   # TODO: Add CSV export
            'export_json_url': '#',  # TODO: Add JSON export
            # New comparison metrics
            'standard_count': len(standard_ratings_dict),
            'guided_count': len(guided_ratings_dict),
            'standard_avg_rating': f"{standard_avg:.2f}" if standard_avg > 0 else "—",
            'guided_avg_rating': f"{guided_avg:.2f}" if guided_avg > 0 else "—",
            'rating_improvement': f"{guided_avg - standard_avg:.2f}" if standard_avg > 0 and guided_avg > 0 else "—",
            'prediction_accuracy': prediction_accuracy or "—",
            'study_phase': 'comparison',
            # Interest rating comparisons
            'standard_interest_count': len(standard_interest),
            'guided_interest_count': len(guided_interest),
            'standard_interest_avg': f"{standard_interest_avg:.2f}" if standard_interest_avg > 0 else "—",
            'guided_interest_avg': f"{guided_interest_avg:.2f}" if guided_interest_avg > 0 else "—",
            'standard_interest_total': f"{standard_interest_total:.1f}" if standard_interest_total > 0 else "—",
            'guided_interest_total': f"{guided_interest_total:.1f}" if guided_interest_total > 0 else "—",
            'interest_improvement': f"{interest_improvement:.2f}" if standard_interest and guided_interest else "—",
            'better_recommendation_phase': better_phase if standard_interest and guided_interest else "—",
        }
        
        return render(request, 'project4/study_results.html', context)
        
    except Exception as e:
        logger.error(f"Error in study_results: {e}")
        return redirect('project4:study_start')

def feedback(request):
    """Handle post-study feedback"""
    if request.method == 'POST':
        try:
            pid = request.session.get('participant_id')
            participant = Participant.objects.get(id=pid)
            
            helpfulness = request.POST.get('helpfulness')
            comments = request.POST.get('comments', '')
            
            Feedback.objects.create(
                participant=participant,
                helpfulness=int(helpfulness) if helpfulness else None,
                comments=comments
            )
            
            return JsonResponse({'status': 'success'})
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return JsonResponse({'status': 'error'})
    
    return JsonResponse({'status': 'invalid_method'})