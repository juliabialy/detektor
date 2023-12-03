import cv2
import numpy as np

def detect_danger_zone(frame, danger_line):
    # Konwertuj obraz do odcieni szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Rozmyj obraz, aby zredukować szum i poprawić detekcję konturów
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binaryzacja obrazu
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Znajdź kontury na obrazie
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Dla każdego konturu znajdź prostokąt obejmujący
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Ignoruj małe kontury, które mogą być szumem
        if cv2.contourArea(contour) < 500:
            continue
        
        # Jeśli obiekt jest poniżej linii niebezpieczeństwa, oznacz go na czerwono
        if y + h > danger_line:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            # W przeciwnym razie oznacz go na zielono
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Narysuj linię niebezpieczeństwa na obrazie
    cv2.line(frame, (0, danger_line), (frame.shape[1], danger_line), (255, 255, 0), 2)

# Ustaw źródło kamery
cap = cv2.VideoCapture(0)

# Ustaw wysokość linii niebezpieczeństwa
danger_line = 300

while cap.isOpened():
    # Pobierz klatkę z kamery
    ret, frame = cap.read()
    if not ret:
        break
    
    # Wywołaj funkcję detekcji
    detect_danger_zone(frame, danger_line)
    
    # Wyświetl wynikowy obraz
    cv2.imshow('Detection', frame)
    
    # Zamknij okno, jeśli naciśnięto klawisz ESC
    if cv2.waitKey(1) == 27:
        break

# Zwolnij zasoby kamery i zamknij okno
cap.release()
cv2.destroyAllWindows()
