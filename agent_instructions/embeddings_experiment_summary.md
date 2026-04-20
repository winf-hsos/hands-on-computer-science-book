# From Manual Word Maps to Real GloVe Embeddings

This summary collects the full teaching flow step by step: a manual introduction to embeddings, a simple vector-based classroom activity, and a transition to real pretrained GloVe embeddings with plots and vector arithmetic.

---

## 1. Learning goal

The idea is to help students understand that:

* words can be represented as vectors
* similar words have similar vectors
* relationships between words can appear as directions in vector space
* real embedding models like GloVe learn these vectors automatically from text

A useful sentence for students is:

> Embeddings are a way to turn meaning into geometry.

---

## 2. Step 1: Manual 2D word map

### Task for students

Place the following words in a 2-dimensional coordinate system.
Words with similar meaning should be placed near each other.
Words with dissimilar meaning should be farther apart.

### Word set (intentionally mixed types)

```text
cat, dog, Paris, Berlin,
king, queen, happy, joyful,
car, bicycle, eat, drink,
fast, slow, summer, winter,
man, woman, red, blue
```

### Why this set is interesting

This set deliberately mixes very different types of meaning:

* animals: `cat`, `dog`
* places: `Paris`, `Berlin`
* people/roles: `king`, `queen`, `man`, `woman`
* emotions: `happy`, `joyful`
* objects: `car`, `bicycle`
* actions: `eat`, `drink`
* properties: `fast`, `slow`
* time concepts: `summer`, `winter`
* colors: `red`, `blue`

👉 This ensures:

* clear clusters emerge
* but also **interesting ambiguities and discussions**

---

## 3. Step 2: Manual vector embeddings

After the 2D activity, students assign each word a value on several scales.
Each word now becomes a vector.

### Recommended dimensions

Rate each word from `0` to `5` on these scales:

1. **Size**: small → large
2. **Living**: not living → living
3. **Human**: not human → human
4. **Mobility**: stationary → moves a lot
5. **Natural**: natural → man-made
6. **Gender**: male → female

---

## ⚠️ Important: This is intentionally imperfect

Some words **cannot be rated cleanly** on these dimensions.

Examples:

* `happy`, `joyful` → no size, no mobility
* `eat`, `drink` → actions, not objects
* `red`, `blue` → abstract properties
* `fast`, `slow` → relative concepts

👉 This is **not a problem** — it is part of the exercise.

---

## 🎯 Teaching moment (very important)

Ask students explicitly:

> Which words were hardest to rate?

Expected answers:

* emotions
* verbs
* abstract properties

Then ask:

> Why was it hard?

👉 Key insight:

> Our chosen dimensions do not capture all types of meaning.

---

## 💡 Core takeaway

> Simple, hand-designed dimensions work for some words, but break down for others.

👉 This directly motivates real embeddings.

---

## 4. Example vectors (for demonstration)

```python
words = {
    "cat":     [2, 5, 1, 4, 5, 2],
    "dog":     [3, 5, 1, 4, 5, 2],
    "car":     [4, 0, 0, 5, 0, 2],
    "bicycle": [3, 0, 0, 5, 0, 2],
    "man":     [3, 5, 5, 3, 5, 0],
    "woman":   [3, 5, 5, 3, 5, 5],
    "king":    [4, 5, 5, 2, 5, 0],
    "queen":   [4, 5, 5, 2, 5, 5],
    # difficult ones (example approximations)
    "happy":   [0, 1, 1, 0, 5, 2],
    "red":     [0, 0, 0, 0, 5, 2],
}
```

👉 Emphasize:

> These values are somewhat arbitrary — and that is exactly the point.

---

## 5. Step 3: Compare words mathematically

### Euclidean distance

```python
import math

def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

print(distance(words["cat"], words["dog"]))
print(distance(words["cat"], words["car"]))
```

### Interpretation

* small distance → words are similar
* large distance → words are less similar

---

## 6. Step 4: Vector arithmetic

### Helper functions

```python
def add(a, b):
    return [x + y for x, y in zip(a, b)]

def sub(a, b):
    return [x - y for x, y in zip(a, b)]
```

### Classic analogy

```python
result = add(sub(words["king"], words["man"]), words["woman"])
```

### Find closest word

```python
def closest_word(query, word_vectors):
    best_word = None
    best_dist = float("inf")
    for word, vector in word_vectors.items():
        d = distance(query, vector)
        if d < best_dist:
            best_dist = d
            best_word = word
    return best_word

print(closest_word(result, words))
```

### Expected result

```text
queen
```

---

## 🎯 Key insight

This works because:

* gender is encoded as a dimension
* vector arithmetic moves along that dimension

---

## 6b. Additional exercise: Food embeddings (3 dimensions)

This exercise mirrors the classic “add a dimension → reveal structure” idea using a food/agriculture example.

### Words

```text
cow, chicken, wheat, apple,
beef, eggs, flour, juice, milk
```

### Step 1 — Two dimensions

1. **Origin**: plant ← → animal
2. **Processing**: raw ← → processed

#### Task

* Place the words in a 2D coordinate system using these two axes.
* Discuss which pairs align naturally.

#### Expected observations

* `cow → beef` aligns with `wheat → flour`
* `apple → juice` aligns with `chicken → eggs` (roughly)

Example analogy:

```text
beef - cow + wheat ≈ flour
```

---

### Step 2 — What is missing?

Ask:

> Do all transformations look equally clear?

Students will notice that something is still not captured well (e.g., liquids vs solids).

---

### Step 3 — Add third dimension

3. **Physical form**: solid ← → liquid

Now re-evaluate the placement of the words.

#### New structure

| Word    | Origin | Processing | Form        |
| ------- | ------ | ---------- | ----------- |
| cow     | animal | raw        | solid       |
| beef    | animal | processed  | solid       |
| wheat   | plant  | raw        | solid       |
| flour   | plant  | processed  | solid       |
| apple   | plant  | raw        | solid       |
| juice   | plant  | processed  | liquid      |
| chicken | animal | raw        | solid       |
| eggs    | animal | processed  | semi-liquid |
| milk    | animal | processed  | liquid      |

---

### Step 4 — New analogies

Now additional analogies become visible:

```text
milk - cow ≈ juice - apple
```

Interpretation:

* cow → milk = animal → liquid product
* apple → juice = plant → liquid product

---

### Key insight

> Adding the right dimension reveals structure that was invisible before.

This mirrors how real embedding models learn many dimensions automatically.

---

## 7. Bridge to real embeddings

Now connect to real models.

Say:

> We tried to manually define meaning with a few dimensions.
> That worked sometimes, but not always.

Then:

> Real embedding models solve this by learning many dimensions automatically from text.

---

## 8. GloVe embeddings

### Install

```bash
pip install numpy matplotlib scikit-learn
```

### Load embeddings

```python
import numpy as np

def load_glove(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            embeddings[parts[0]] = np.array(parts[1:], dtype=float)
    return embeddings

emb = load_glove("glove.6B.50d.txt")
```

---

## 9. Cosine similarity

```python
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 10. Real analogy

```python
def analogy(a, b, c):
    query = emb[a] - emb[b] + emb[c]
    best_word = None
    best_score = -1
    for w, v in emb.items():
        if w in {a, b, c}:
            continue
        score = cosine(query, v)
        if score > best_score:
            best_score = score
            best_word = w
    return best_word

print(analogy("king", "man", "woman"))
```

---

## 11. Visualizing with arrows (most intuitive)

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

words = [
    "man", "woman",
    "king", "queen",
    "boy", "girl",
    "father", "mother"
]

pairs = [
    ("man", "woman"),
    ("king", "queen"),
    ("boy", "girl"),
    ("father", "mother"),
]

X = np.array([emb[w] for w in words])
X2 = PCA(n_components=2).fit_transform(X)
coords = {w: X2[i] for i, w in enumerate(words)}

plt.figure(figsize=(8, 6))

for w, (x, y) in coords.items():
    plt.scatter(x, y)
    plt.text(x + 0.02, y + 0.02, w)

for a, b in pairs:
    x1, y1 = coords[a]
    x2, y2 = coords[b]
    plt.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.03)

plt.axhline(0)
plt.axvline(0)
plt.title("Semantic directions (GloVe)")
plt.show()
```

---

## 🎯 Final takeaway

```text
manual intuition → imperfect vectors → limitations → learned embeddings
```

---

## 🧠 Closing sentence

> Words are points, relationships are directions, and meaning lives in geometry.
