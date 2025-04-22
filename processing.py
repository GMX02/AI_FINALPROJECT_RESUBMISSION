# processing.py (dummy implementation)

def get_audio_info(file_path):
    # pretend to extract length and sample rate
    return {
        'length': 132.4,  # seconds
        'sample_rate': 44100  # Hz
    }

def detect_gunshot(file_path):
    # fake detection result
    return {
        'presence': True,  # or False
        'confidence': 87.5  # percent
    }

def locate_gunshots(file_path):
    # fake time stamps in seconds
    return [12.3, 58.9, 102.7]

def categorize_firearm(file_path):
    # dummy classification
    return {
        'firearm': 'Glock 17',
        'caliber': '9mm',
        'match_confidence': 78.9
    }
