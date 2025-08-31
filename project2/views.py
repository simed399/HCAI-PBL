from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from .ml.experiment   import run_experiment as run_experiment_sim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .utils.dataset import load_imdb
from .ml.pipeline import make_pipeline, train_and_save, load_pipeline
from .ml.active_learning import (
    uncertainty_sampling, margin_sampling, entropy_sampling, committee_sampling
)
from .models import HumanLabelingSession, HumanLabeledSample, SessionProgress
from django.db import models
import os
import json
import time
import threading
import uuid
import pickle
import numpy as np
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model_full.pkl')

# Progress tracking with real-time experiment data
progress_data = {
    'task1': {'progress': 0, 'status': 'idle', 'message': ''},
    'task2': {'progress': 0, 'status': 'idle', 'message': ''},
    'task3': {'progress': 0, 'status': 'idle', 'message': '', 'experiment_data': []}
}

def index(request):
    return render(request, 'project2/index.html')

def get_progress(request, task):
    """Get progress for a specific task"""
    if task in progress_data:
        return JsonResponse(progress_data[task])
    return JsonResponse({'error': 'Task not found'})

def start_training(request):
    """Start the training process in background"""
    # Reset progress data
    progress_data['task1'] = {'progress': 0, 'status': 'idle', 'message': ''}
    
    # Initialize progress
    progress_data['task1']['status'] = 'running'
    progress_data['task1']['progress'] = 10
    progress_data['task1']['message'] = 'Loading IMDB training dataset...'
    
    def run_training():
        try:
            X_train, y_train = load_imdb('train')
            X_test,  y_test  = load_imdb('test')
            
            progress_data['task1']['progress'] = 30
            progress_data['task1']['message'] = 'Dataset loaded. Checking for existing model...'

            # Always retrain to avoid model state issues
            progress_data['task1']['progress'] = 40
            progress_data['task1']['message'] = 'Creating fresh pipeline...'
            pipeline = make_pipeline()
            progress_data['task1']['progress'] = 50
            progress_data['task1']['message'] = f'Training model on {len(X_train)} training samples...'
            pipeline = train_and_save(pipeline, X_train, y_train, MODEL_PATH)
            progress_data['task1']['progress'] = 80
            progress_data['task1']['message'] = 'Training complete. Evaluating...'

            # Evaluate
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            progress_data['task1']['progress'] = 100
            progress_data['task1']['status'] = 'complete'
            progress_data['task1']['message'] = f'Complete! Test accuracy: {acc:.2f}%'
            
        except Exception as e:
            progress_data['task1']['status'] = 'error'
            progress_data['task1']['message'] = f'Error: {str(e)}'
    
    # Start training in background thread
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return JsonResponse({'status': 'started'})

def train_full(request):
    """Show training results or start training"""
    # Check if we have a completed training
    if progress_data['task1']['status'] == 'complete':
        # Extract accuracy from message
        message = progress_data['task1']['message']
        import re
        accuracy_match = re.search(r'(\d+\.\d+)%', message) if message else None
        accuracy = float(accuracy_match.group(1)) if accuracy_match else None
        return render(request, 'project2/train.html', {'accuracy': accuracy, 'show_progress': False})
    else:
        return render(request, 'project2/train.html', {'accuracy': None, 'show_progress': True})


@csrf_exempt
def upload_model(request):
    """Handle pre-trained model upload and evaluation"""
    if request.method == 'POST' and request.FILES.get('model_file'):
        try:
            model_file = request.FILES['model_file']
            
            # Validate file extension
            if not model_file.name.endswith(('.pkl', '.pickle')):
                return JsonResponse({
                    'status': 'error', 
                    'message': 'Invalid file format. Please upload a .pkl or .pickle file.'
                })
            
            # Load the uploaded model
            model_content = model_file.read()
            pipeline = pickle.loads(model_content)
            
            # Validate that it's a proper pipeline
            if not hasattr(pipeline, 'predict') or not hasattr(pipeline, 'fit'):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid model file. The uploaded file does not contain a valid sklearn pipeline.'
                })
            
            # Load test data for evaluation
            X_test, y_test = load_imdb('test')
            
            # Evaluate the uploaded model
            try:
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
                
                # Save the uploaded model as the current model
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(pipeline, f)
                
                # Update progress data to reflect the uploaded model
                progress_data['task1']['status'] = 'complete'
                progress_data['task1']['progress'] = 100
                progress_data['task1']['message'] = f'Uploaded model evaluated! Test accuracy: {acc:.2f}%'
                
                return JsonResponse({
                    'status': 'success',
                    'accuracy': f'{acc:.2f}',
                    'message': 'Model uploaded and evaluated successfully!'
                })
                
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Error evaluating model: {str(e)}. Make sure the model is compatible with the IMDB dataset.'
                })
                
        except pickle.UnpicklingError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid pickle file. Please ensure the file is a valid pickled sklearn pipeline.'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error processing uploaded file: {str(e)}'
            })
    
    return JsonResponse({'status': 'error', 'message': 'No file uploaded or invalid request method.'})


def active_learning(request):
    """Render form to choose strategy & number of queries."""
    return render(request, 'project2/active_learning.html')

def run_active_learning(request):
    """
    Simulate one pool‐based AL iteration:
    1. Query n samples by chosen strategy
    2. “Label” them via ground-truth
    3. Retrain on just those points
    4. Report test accuracy + show selected samples
    """
    # Capture form data before threading
    strategy = request.POST['strategy']
    n_q = int(request.POST['n_queries'])
    
    # Reset progress data
    progress_data['task2'] = {'progress': 0, 'status': 'idle', 'message': ''}
    
    # Initialize progress
    progress_data['task2']['status'] = 'running'
    progress_data['task2']['progress'] = 10
    progress_data['task2']['message'] = 'Loading IMDB dataset...'
    
    def run_al():
        import numpy as np  # Ensure numpy is available in thread scope
        try:
            # load pool + test with original text for display
            X_pool, X_pool_original, y_pool = load_imdb('train', return_original=True)
            X_test, y_test = load_imdb('test')
            
            progress_data['task2']['progress'] = 20
            progress_data['task2']['message'] = f'Dataset loaded. Using {strategy} sampling strategy...'

            # start from scratch with proper active learning initialization
            progress_data['task2']['progress'] = 30
            progress_data['task2']['message'] = 'Creating initial model with minimal bootstrap data...'
            
            pipeline = make_pipeline()
            
            # Use proper active learning initialization - one sample per class like Task 3
            unique_classes = np.unique(y_pool)
            labelled_idxs = []
            
            # Start with one sample from each class for proper bootstrap
            for class_label in unique_classes:
                class_indices = np.where(np.array(y_pool) == class_label)[0]
                idx = np.random.choice(class_indices)
                labelled_idxs.append(idx)
            
            # Fit pipeline with samples from both classes  
            X_initial = [X_pool[i] for i in labelled_idxs]
            y_initial = [y_pool[i] for i in labelled_idxs]
            pipeline.fit(X_initial, y_initial)
            
            progress_data['task2']['progress'] = 40
            progress_data['task2']['message'] = f'Bootstrap complete with {len(labelled_idxs)} samples. Starting iterative selection...'

            # Iteratively select samples (like Task 3 but simplified)
            selected_samples = []
            
            for iteration in range(n_q):
                progress_data['task2']['progress'] = 40 + int((iteration / n_q) * 40)  # 40-80% range
                progress_data['task2']['message'] = f'Iteration {iteration + 1}/{n_q}: Selecting most informative sample...'
                
                # Build pool of unlabeled samples
                unlabeled_idxs = [i for i in range(len(X_pool)) if i not in labelled_idxs]
                
                if len(unlabeled_idxs) == 0:
                    break
                    
                X_unlabeled = [X_pool[i] for i in unlabeled_idxs]
                
                # Select next sample using chosen strategy
                if strategy == 'uncertainty':
                    next_idx_in_pool = uncertainty_sampling(pipeline, X_unlabeled, 1)[0]
                elif strategy == 'margin':
                    next_idx_in_pool = margin_sampling(pipeline, X_unlabeled, 1)[0]
                elif strategy == 'committee':
                    next_idx_in_pool = committee_sampling(pipeline, X_unlabeled, 1)[0]
                else:
                    next_idx_in_pool = entropy_sampling(pipeline, X_unlabeled, 1)[0]
                
                # Map back to original pool index
                next_idx = unlabeled_idxs[next_idx_in_pool]
                labelled_idxs.append(next_idx)
                selected_samples.append(X_pool_original[next_idx])  # Store original text for display
                
                # Retrain with updated dataset (iterative improvement)
                X_lab = [X_pool[i] for i in labelled_idxs]
                y_lab = [y_pool[i] for i in labelled_idxs]
                pipeline.fit(X_lab, y_lab)
            
            progress_data['task2']['progress'] = 85
            progress_data['task2']['message'] = f'Selected {len(selected_samples)} samples iteratively. Final evaluation...'

            # Final evaluation
            acc = accuracy_score(y_test, pipeline.predict(X_test))
            acc_percentage = acc * 100  # Convert to percentage
            
            progress_data['task2']['progress'] = 95
            progress_data['task2']['message'] = f'Active learning complete! Total samples used: {len(labelled_idxs)}'
            
            progress_data['task2']['progress'] = 100
            progress_data['task2']['status'] = 'complete'
            progress_data['task2']['message'] = f'Complete! Test accuracy: {acc_percentage:.2f}% (trained on {len(labelled_idxs)} samples)'

            # Store results for later retrieval
            progress_data['task2']['results'] = {
                'strategy': strategy,
                'n_queries': n_q,
                'accuracy': acc_percentage,  # Store as percentage
                'selected': selected_samples,  # Use collected original text samples
                'total_samples': len(labelled_idxs),
            }
            
        except Exception as e:
            progress_data['task2']['status'] = 'error'
            progress_data['task2']['message'] = f'Error: {str(e)}'
    
    # Start AL in background thread
    thread = threading.Thread(target=run_al)
    thread.start()
    
    return JsonResponse({'status': 'started'})

def get_al_results(request):
    """Get active learning results after completion"""
    if progress_data['task2']['status'] == 'complete' and 'results' in progress_data['task2']:
        results = progress_data['task2']['results']
        return render(request, 'project2/active_result.html', results)
    else:
        return JsonResponse({'status': 'not_ready'})

def get_experiment_data(request):
    """Get real-time experiment data for visualization"""
    if 'experiment_data' in progress_data['task3']:
        return JsonResponse({
            'status': progress_data['task3']['status'],
            'data': progress_data['task3']['experiment_data'],
            'progress': progress_data['task3']['progress'],
            'message': progress_data['task3']['message']
        })
    else:
        return JsonResponse({'status': 'no_data', 'data': []})

def get_exp_results(request):
    """Get experiment results after completion"""
    if progress_data['task3']['status'] == 'complete' and 'results' in progress_data['task3']:
        results = progress_data['task3']['results']
        return render(request, 'project2/experiment_result.html', results)
    else:
        return JsonResponse({'status': 'not_ready'})

def show_experiment_live(request):
    """Show live experiment visualization page"""
    # Force reload to register URL pattern - refresh template cache - reload 3
    return render(request, 'project2/experiment_live.html')

def experiment(request):
    """Form to choose strategy & budget for the full experiment."""
    return render(request, 'project2/experiment.html')

def run_experiment(request):
    """Run the multi–step simulation with stopping conditions, plot accuracy vs. #labels, and display."""
    # Capture form data before threading
    strategy = request.POST['strategy']
    budget = int(request.POST['budget'])
    target_accuracy = request.POST.get('target_accuracy')
    
    # Get plateau detection parameters
    plateau_patience = int(request.POST.get('plateau_patience', 5))
    plateau_threshold = float(request.POST.get('plateau_threshold', 0.5))
    # Handle checkbox: checked = 'on', unchecked = None
    enable_plateau_detection = 'enable_plateau_detection' in request.POST
    
    # Convert target accuracy to float if provided
    target_acc_float = None
    if target_accuracy and target_accuracy.strip():
        try:
            target_acc_float = float(target_accuracy) / 100.0  # Convert percentage to decimal
        except (ValueError, TypeError):
            target_acc_float = None
    
    # Convert plateau threshold from percentage to decimal
    plateau_threshold_decimal = plateau_threshold / 100.0
    
    # Reset progress data with experiment data tracking
    progress_data['task3'] = {'progress': 0, 'status': 'idle', 'message': '', 'experiment_data': []}
    
    # Initialize progress
    progress_data['task3']['status'] = 'running'
    progress_data['task3']['progress'] = 10
    progress_data['task3']['message'] = 'Loading IMDB dataset...'
    
    def run_exp():
        try:
            # load data
            X_pool, y_pool = load_imdb('train')
            X_test, y_test = load_imdb('test')
            
            progress_data['task3']['progress'] = 30
            progress_data['task3']['message'] = f'Dataset loaded. Running {strategy} experiment...'

            # Define progress callback for real-time updates
            def progress_callback(data):
                progress_data['task3']['experiment_data'].append({
                    'x': data['samples_used'],
                    'y': data['accuracy'] * 100,  # Convert to percentage
                    'iteration': data['iteration']
                })
                # Update progress based on samples used
                progress_percent = 30 + (data['samples_used'] / budget) * 40  # 30-70% range
                progress_data['task3']['progress'] = min(int(progress_percent), 70)
                progress_data['task3']['message'] = f'Iteration {data["iteration"]}: {data["samples_used"]} samples, {data["accuracy"]*100:.1f}% accuracy'

            # run sim with stopping conditions and real-time callback
            experiment_result = run_experiment_sim(X_pool, y_pool, X_test, y_test, strategy, budget, 
                                                  target_acc_float, progress_callback, 
                                                  plateau_patience, plateau_threshold_decimal, enable_plateau_detection)
            
            # Extract results
            accuracies = experiment_result['accuracies']
            stop_reason = experiment_result['stop_reason']
            final_accuracy = experiment_result['final_accuracy']
            samples_used = experiment_result['samples_used']
            
            progress_data['task3']['progress'] = 70
            progress_data['task3']['message'] = f'Experiment stopped: {stop_reason.replace("_", " ").title()}. Generating visualization...'

            # plot to media with dark theme
            media_subdir = 'project2/experiments'
            media_dir    = os.path.join(settings.MEDIA_ROOT, media_subdir)
            os.makedirs(media_dir, exist_ok=True)
            plot_path    = os.path.join(media_dir, 'accuracy_vs_budget.png')

            # Set dark theme for matplotlib
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f1220')
            ax.set_facecolor('#171a2b')
            
            # Plot with custom colors matching the UI theme
            ax.plot(range(len(accuracies)), [acc*100 for acc in accuracies], 
                   color='#6aa2ff', linewidth=3, marker='o', markersize=8, 
                   markerfacecolor='#6effc6', markeredgecolor='#6aa2ff', markeredgewidth=2)
            
            ax.set_xlabel('Number of labeled samples', color='#a9b0c3', fontsize=12, fontweight='bold')
            ax.set_ylabel('Test accuracy (%)', color='#a9b0c3', fontsize=12, fontweight='bold')
            ax.set_title(f'{strategy.title()} Sampling - Stopped: {stop_reason.replace("_", " ").title()}', 
                        color='#e9ecf2', fontsize=14, fontweight='bold', pad=20)
            
            # Customize grid and ticks
            ax.grid(True, alpha=0.3, color='#23263a')
            ax.tick_params(colors='#a9b0c3', labelsize=10)
            
            # Add target accuracy line if specified
            if target_acc_float:
                ax.axhline(y=target_acc_float*100, color='#ff7676', linestyle='--', alpha=0.8, linewidth=2, 
                          label=f'Target: {target_acc_float*100:.1f}%')
                ax.legend(facecolor='#171a2b', edgecolor='#23263a', labelcolor='#e9ecf2')
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_color('#23263a')
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0f1220')
            plt.close()
            
            progress_data['task3']['progress'] = 90
            progress_data['task3']['message'] = 'Plot generated. Finalizing results...'

            image_url = settings.MEDIA_URL + f'{media_subdir}/accuracy_vs_budget.png'
            
            progress_data['task3']['progress'] = 100
            progress_data['task3']['status'] = 'complete'
            progress_data['task3']['message'] = f'Complete! {stop_reason.replace("_", " ").title()} - Final accuracy: {final_accuracy*100:.1f}%'
            
            # Store results for later retrieval
            progress_data['task3']['results'] = {
                'strategy': strategy,
                'budget': budget,
                'target_accuracy': target_accuracy,
                'plateau_patience': plateau_patience,
                'plateau_threshold': plateau_threshold,
                'final_accuracy': final_accuracy * 100,  # Convert to percentage
                'samples_used': samples_used,
                'stop_reason': stop_reason,
                'stop_reason_display': stop_reason.replace('_', ' ').title(),
                'image_url': image_url,
            }
            
        except Exception as e:
            progress_data['task3']['status'] = 'error'
            progress_data['task3']['message'] = f'Error: {str(e)}'
    
    # Start experiment in background thread
    thread = threading.Thread(target=run_exp)
    thread.start()
    
    return JsonResponse({'status': 'started'})


# ============================================================================
# HUMAN LABELING INTERFACE
# ============================================================================

def human_labeling_setup(request):
    """Setup page for human labeling sessions"""
    return render(request, 'project2/human_labeling_setup.html')


def start_human_labeling(request):
    """Start a new human labeling session"""
    if request.method == 'POST':
        # Get form data
        strategy = request.POST['strategy']
        max_samples = int(request.POST.get('max_samples', 50))
        target_accuracy = request.POST.get('target_accuracy')
        batch_size = int(request.POST.get('batch_size', 1))
        
        # Get plateau detection parameters
        plateau_patience = int(request.POST.get('plateau_patience', 5))
        plateau_threshold = float(request.POST.get('plateau_threshold', 0.5))
        
        # Create unique session ID
        session_id = str(uuid.uuid4())[:8]
        
        try:
            # Load dataset with original text for human display
            X_pool, X_pool_original, y_pool = load_imdb('train', return_original=True)
            X_test, y_test = load_imdb('test')
            
            # Create session
            session = HumanLabelingSession.objects.create(
                session_id=session_id,
                strategy=strategy,
                total_samples=max_samples,
                batch_size=batch_size,
                max_samples=max_samples,
                target_accuracy=float(target_accuracy) if target_accuracy else None
            )
            
            # Start with initial model (small subset for bootstrapping)
            initial_size = min(100, len(X_pool))
            X_initial = X_pool[:initial_size]
            y_initial = y_pool[:initial_size]
            
            pipeline = make_pipeline()
            pipeline.fit(X_initial, y_initial)
            
            # Get initial accuracy
            initial_acc = accuracy_score(y_test, pipeline.predict(X_test))
            session.initial_accuracy = initial_acc
            session.current_accuracy = initial_acc
            session.save()
            
            # Record initial progress
            SessionProgress.objects.create(
                session=session,
                samples_labeled=0,
                test_accuracy=initial_acc
            )
            
            # Remove initial samples from pool
            X_pool = X_pool[initial_size:]
            X_pool_original = X_pool_original[initial_size:]
            y_pool = y_pool[initial_size:]
            
            # Select first batch of samples using chosen strategy
            if strategy == 'uncertainty':
                selected_indices = uncertainty_sampling(pipeline, X_pool, batch_size)
            elif strategy == 'margin':
                selected_indices = margin_sampling(pipeline, X_pool, batch_size)
            elif strategy == 'committee':
                selected_indices = committee_sampling(pipeline, X_pool, batch_size)
            else:  # entropy
                selected_indices = entropy_sampling(pipeline, X_pool, batch_size)
            
            # Store selected samples for labeling
            for i, idx in enumerate(selected_indices):
                uncertainty_score = None
                if hasattr(pipeline, 'predict_proba'):
                    probs = pipeline.predict_proba([X_pool[idx]])[0]
                    uncertainty_score = 1 - max(probs)
                
                HumanLabeledSample.objects.create(
                    session=session,
                    sample_text=X_pool[idx],  # Preprocessed text for ML
                    sample_original_text=X_pool_original[idx],  # Original text for display
                    sample_index=initial_size + idx,  # Adjust for removed initial samples
                    ground_truth_label=y_pool[idx],
                    uncertainty_score=uncertainty_score,
                    selection_order=i + 1
                )
            
            return redirect('project2:human_labeling_session', session_id=session_id)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return redirect('project2:human_labeling_setup')


def human_labeling_session(request, session_id):
    """Main human labeling interface"""
    session = get_object_or_404(HumanLabelingSession, session_id=session_id)
    
    # Get next unlabeled sample
    next_sample = session.samples.filter(human_label__isnull=True).first()
    
    if not next_sample or session.is_session_complete():
        # Session is complete, redirect to results
        session.is_completed = True
        session.completed_at = timezone.now()
        session.save()
        return redirect('project2:human_labeling_results', session_id=session_id)
    
    # Get session progress
    labeled_count = session.samples.filter(human_label__isnull=False).count()
    total_count = session.samples.count()
    
    context = {
        'session': session,
        'sample': next_sample,
        'labeled_count': labeled_count,
        'total_count': total_count,
        'progress_percentage': (labeled_count / total_count * 100) if total_count > 0 else 0,
    }
    
    return render(request, 'project2/human_labeling_session.html', context)


@csrf_exempt
def submit_human_label(request, session_id, sample_id):
    """Submit a human label for a sample"""
    if request.method == 'POST':
        session = get_object_or_404(HumanLabelingSession, session_id=session_id)
        sample = get_object_or_404(HumanLabeledSample, id=sample_id, session=session)
        
        # Get form data
        data = json.loads(request.body) if request.content_type == 'application/json' else request.POST
        human_label = int(data['label'])
        confidence = data.get('confidence')
        labeling_time = data.get('labeling_time')
        
        # Update sample
        sample.human_label = human_label
        sample.confidence = confidence
        sample.labeling_time_seconds = labeling_time
        sample.labeled_at = timezone.now()
        sample.save()
        
        # Update session progress
        session.labeled_samples = session.samples.filter(human_label__isnull=False).count()
        
        # Retrain model and update accuracy
        try:
            # Get all labeled samples
            labeled_samples = session.samples.filter(human_label__isnull=False)
            
            if labeled_samples.count() >= 2:  # Need at least 2 samples
                X_labeled = [s.sample_text for s in labeled_samples]
                y_labeled = [s.human_label for s in labeled_samples]
                
                # Retrain model
                pipeline = make_pipeline()
                pipeline.fit(X_labeled, y_labeled)
                
                # Evaluate on test set
                X_test, y_test = load_imdb('test')
                y_pred = pipeline.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Update session
                session.current_accuracy = accuracy
                session.save()
                
                # Record progress
                SessionProgress.objects.create(
                    session=session,
                    samples_labeled=session.labeled_samples,
                    test_accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1
                )
                
                # Check if we need to select more samples
                unlabeled_count = session.samples.filter(human_label__isnull=True).count()
                
                if (unlabeled_count == 0 and 
                    session.labeled_samples < session.max_samples and 
                    not session.is_session_complete()):
                    
                    # Select next batch of samples
                    X_pool, X_pool_original, y_pool = load_imdb('train', return_original=True)
                    
                    # Get indices of already selected samples
                    selected_indices = set(session.samples.values_list('sample_index', flat=True))
                    
                    # Create pool of remaining samples
                    remaining_X = []
                    remaining_X_original = []
                    remaining_y = []
                    remaining_indices = []
                    
                    for i, (x, x_orig, y) in enumerate(zip(X_pool, X_pool_original, y_pool)):
                        if i not in selected_indices:
                            remaining_X.append(x)
                            remaining_X_original.append(x_orig)
                            remaining_y.append(y)
                            remaining_indices.append(i)
                    
                    if remaining_X:
                        # Select new samples
                        batch_size = min(session.batch_size, session.max_samples - session.labeled_samples.count())
                        
                        if session.strategy == 'uncertainty':
                            new_indices = uncertainty_sampling(pipeline, remaining_X, batch_size)
                        elif session.strategy == 'margin':
                            new_indices = margin_sampling(pipeline, remaining_X, batch_size)
                        elif session.strategy == 'committee':
                            new_indices = committee_sampling(pipeline, remaining_X, batch_size)
                        else:  # entropy
                            new_indices = entropy_sampling(pipeline, remaining_X, batch_size)
                        
                        # Add new samples to session
                        current_order = session.samples.count()
                        for i, idx in enumerate(new_indices):
                            original_idx = remaining_indices[idx]
                            uncertainty_score = None
                            if hasattr(pipeline, 'predict_proba'):
                                probs = pipeline.predict_proba([remaining_X[idx]])[0]
                                uncertainty_score = 1 - max(probs)
                            
                            HumanLabeledSample.objects.create(
                                session=session,
                                sample_text=remaining_X[idx],  # Preprocessed text for ML
                                sample_original_text=remaining_X_original[idx],  # Original text for display
                                sample_index=original_idx,
                                ground_truth_label=remaining_y[idx],
                                uncertainty_score=uncertainty_score,
                                selection_order=current_order + i + 1
                            )
        
        except Exception as e:
            print(f"Error in retraining: {e}")
        
        return JsonResponse({
            'success': True,
            'session_complete': session.is_session_complete(),
            'next_url': f"/project2/human-labeling/session/{session_id}/" if not session.is_session_complete() else f"/project2/human-labeling/results/{session_id}/"
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


def human_labeling_results(request, session_id):
    """Display results of human labeling session"""
    session = get_object_or_404(HumanLabelingSession, session_id=session_id)
    
    # Get all progress snapshots
    progress_snapshots = session.progress_snapshots.all()
    
    # Calculate statistics
    labeled_samples = session.samples.filter(human_label__isnull=False)
    total_labeled = labeled_samples.count()
    
    # Accuracy of human labels vs ground truth
    correct_labels = sum(1 for sample in labeled_samples if sample.human_label == sample.ground_truth_label)
    human_accuracy = (correct_labels / total_labeled * 100) if total_labeled > 0 else 0
    
    # Average labeling time
    labeled_with_time = labeled_samples.exclude(labeling_time_seconds__isnull=True)
    avg_time = labeled_with_time.aggregate(avg_time=models.Avg('labeling_time_seconds'))['avg_time'] if labeled_with_time.exists() else None
    
    # Prepare data for learning curve
    learning_curve_data = [
        [p.samples_labeled, p.test_accuracy * 100] 
        for p in progress_snapshots
    ]
    
    context = {
        'session': session,
        'total_labeled': total_labeled,
        'human_accuracy': human_accuracy,
        'avg_labeling_time': avg_time,
        'final_accuracy': session.current_accuracy * 100 if session.current_accuracy else 0,
        'accuracy_improvement': (session.current_accuracy - session.initial_accuracy) * 100 if session.current_accuracy and session.initial_accuracy else 0,
        'learning_curve_data': json.dumps(learning_curve_data),
        'labeled_samples': labeled_samples.order_by('selection_order')[:10],  # Show first 10
    }
    
    return render(request, 'project2/human_labeling_results.html', context)


def get_session_progress(request, session_id):
    """Get progress data for a human labeling session"""
    session = get_object_or_404(HumanLabelingSession, session_id=session_id)
    
    labeled_count = session.samples.filter(human_label__isnull=False).count()
    total_count = session.samples.count()
    
    return JsonResponse({
        'session_id': session_id,
        'labeled_count': labeled_count,
        'total_count': total_count,
        'progress_percentage': session.progress_percentage,
        'current_accuracy': session.current_accuracy * 100 if session.current_accuracy else 0,
        'is_complete': session.is_session_complete(),
        'strategy': session.strategy,
    })