# Projekt 1.9: Równoległy algorytm klasteryzacji k-means (CUDA)

Dokumentacja wstępna (plan implementacji) oraz specyfikacja techniczna projektu realizowanego w ramach przedmiotu **Procesory graficzne w zastosowaniach obliczeniowych** (Wydział MiNI PW).

---

## 1. Cel projektu i założenia algorytmiczne

Celem projektu jest implementacja oraz porównanie wydajności równoległego algorytmu grupowania **k-means** w środowisku NVIDIA CUDA dla dużych zbiorów danych wielowymiarowych, wraz z sekwencyjną implementacją referencyjną na CPU.

### Parametry i reguły algorytmu
* **Skala danych:** liczba punktów $N \le 50\,000\,000$, liczba wymiarów $d \in [1, 20]$, liczba klastrów $k \in [2, 20]$.
* **Precyzja obliczeń:** liczby zmiennoprzecinkowe podwójnej precyzji (`double`, 8 bajtów).
* **Inicjalizacja centroidów:** obowiązkowy wybór **pierwszych $k$ punktów** ze zbioru wejściowego.
* **Przypisanie do klastra:** każdy punkt przypisywany jest do najbliższego centroidu według kwadratu odległości euklidesowej (bez kosztownego pierwiastkowania); w przypadku remisu wybierany jest klaster o mniejszym indeksie.
* **Warunek stopu:** brak zmiany przynależności do klastrów dla wszystkich $N$ punktów w bieżącej iteracji **lub** osiągnięcie limitu $100$ iteracji.
* **Obsługa pustych klastrów:** jeżeli do klastra $c$ nie trafi żaden punkt ($\text{count}[c] = 0$), centroid zachowuje swoją pozycję z poprzedniej iteracji (zabezpieczenie przed dzieleniem przez zero).

---

## 2. Model danych i optymalizacja pamięci

### Wczytywanie danych i transpozycja AoS $\rightarrow$ SoA
W obu formatach wejściowych (`txt` oraz `bin`) współrzędne pojedynczego punktu leżą w pliku obok siebie (układ AoS: $x_0, y_0, \dots, x_1, y_1, \dots$). Aby uniknąć duplikowania danych w pamięci RAM (przy $N = 50\text{ mln}$ i $d = 20$ zbiór zajmuje $8\text{ GB}$), transpozycja odbywa się **w locie podczas czytania pliku**:
* Dla formatu `txt` współrzędne czytane są funkcją `fscanf` i od razu zapisywane pod docelowym indeksem SoA.
* Dla formatu `bin` plik czytany jest funkcją `fread` porcjami (np. po 65 536 punktów = $2^{16}$) do niewielkiego bufora pomocniczego, z którego współrzędne od razu trafiają do tablicy SoA.

### Układ tablic i koalescencja odczytów na GPU
Wszystkie punkty przechowywane są w jednej płaskiej tablicy jednowymiarowej `data` (wymaga pojedynczego wywołania `cudaMalloc`), a rozmiary w bajtach obliczane są wyłącznie w 64-bitowym typie `size_t`:
* Współrzędna `dim` punktu `idx` znajduje się pod indeksem:
  $$\text{index} = \text{dim} \cdot N + \text{idx}$$
* Ponieważ w każdym kroku sąsiednie wątki w warpie ($\text{idx}, \text{idx}+1, \dots, \text{idx}+31$) przetwarzają sąsiednie punkty, czytają one w tym samym takcie sąsiadujące 8-bajtowe słowa `double` w VRAM, zapewniając pełną koalescencję transakcji pamięciowych.
* Pozostałe tablice w pamięci globalnej: `labels[N]` (aktualne przypisania punktów), `sums[k * d]` oraz `count[k]` (akumulatory do wyznaczania nowych środków klastrów).

### Wykorzystanie pamięci stałej (`__constant__`)
Przy maksymalnych parametrach zadania ($k = 20, d = 20$, typ `double`) tablica aktualnych centroidów zajmuje zaledwie **$20 \times 20 \times 8\text{ B} = 3200\text{ B}$** (ok. $3,1\text{ KB}$), więc mieści się z dużym zapasem w $64\text{ KB}$ pamięci stałej GPU.

Na początku każdej iteracji aktualne pozycje centroidów kopiowane są do `__constant__` (funkcją `cudaMemcpyToSymbol`) z dwóch powodów:
* **Sprzętowy broadcast w obrębie warpa:** Wszystkie wątki w warpie porównują swoje punkty z tymi samymi centroidami w dokładnie tej samej kolejności. Ponieważ w danym kroku wszystkie 32 wątki pytają o ten sam adres, Constant Cache pobiera wartość raz i rozsyła ją jednocześnie do całego warpa (*broadcast*) z opóźnieniem zbliżonym do odczytu z rejestrów.
* **Ochrona pamięci podręcznej L1/L2:** Przy $50\text{ mln}$ punktów strumień danych nieustannie przepływa przez pamięć podręczną L1/L2. Umieszczenie centroidów w `__constant__` daje im odizolowany tor pamięci podręcznej, którego miliony punktów nigdy nie nadpiszą.

---

## 3. Warianty obliczeniowe (`cpu`, `gpu1`, `gpu2`)

Program realizuje trzy tryby obliczeniowe wybierane parametrem linii poleceń (`computation_method`):

1. **`cpu` (Wersja sekwencyjna — referencyjna):**
   * Jednowątkowa implementacja algorytmu k-means w czystym C/C++ bez zewnętrznych bibliotek.
   * Służy do weryfikacji identyczności wyników z obiema metodami GPU oraz stanowi punkt odniesienia do wyznaczenia przyspieszenia (*speedup*) uzyskanego na GPU.

2. **`gpu1` (Czysta CUDA — ręczne kernele i redukcja blokowa w `__shared__`):**
   * Niskopoziomowa implementacja napisana bezpośrednio w CUDA C++, zaprojektowana tak, aby w jednym głównym przebiegu po 8-gigabajtowej tablicy punktów zarówno przypisać punkty do klastrów, jak i zebrać sumy pod nowe centroidy (bez dwukrotnego czytania danych z VRAM-u):
     * **Stała liczba bloków zamiast setek tysięcy małych bloków:** Zamiast tworzyć ok. 200 tysięcy bloków (po jednym na każde 256 punktów), uruchamiamy stałą, mniejszą siatkę (np. 1024 bloki po 256 wątków) — tak dobraną, aby w 100% zająć wszystkie jednostki obliczeniowe karty. Każdy wątek przetwarza swój punkt startowy, a następnie w pętli przeskakuje o łączną liczbę wszystkich wątków na karcie, aż przerobi cały zbiór. Dzięki temu karta nie traci czasu na ciągłe tworzenie i zamykanie tysięcy bloków, a każdy blok przemiela u siebie tysiące punktów i wysyła wynik do wolnej pamięci globalnej tylko raz na sam koniec.
     * **Wyznaczanie `best_c` z pamięci `__constant__` i zliczanie zmian:** W każdym kroku pętli wątek pobiera współrzędne swojego punktu koalescencyjnie z tablicy SoA i porównuje je ze wszystkimi $k$ centroidami leżącymi w pamięci `__constant__` (korzystając ze sprzętowego rozgłosu *broadcast* dla całego warpa). Liczy kwadrat odległości euklidesowej, wybiera najbliższy klaster `best_c` (przy remisie bierze mniejszy indeks) i zapisuje go do `labels[idx]`. Każdy wątek zlicza swoje zmiany przynależności lokalnie w prywatnej zmiennej (w rejestrze), a po zakończeniu pętli dopisuje je przez `atomicAdd` do jednego licznika bloku w pamięci współdzielonej (`__shared__ int s_changed`). Po barierze `__syncthreads()` tylko jeden wątek z bloku dodaje wynik bloku do globalnego licznika zmian.
     * **Wyliczanie nowych centroidów (średniej arytmetycznej) między blokami:** Pojedynczy blok widzi tylko wycinek całego zbioru, więc nie może od razu dzielić przez liczbę punktów. Zamiast tego każdy blok ma w swojej szybkiej pamięci współdzielonej lokalny „brudnopis”: `__shared__ double s_sums[20][21]` oraz liczniki `__shared__ int s_count[20]` (dodatkowa 21. kolumna to padding, który przesuwa adresy między bankami pamięci i eliminuje konflikty banków). Wątki dopisują współrzędne i podbijają liczniki lokalnie w `__shared__`, a dopiero po zakończeniu pętli i barierze `__syncthreads()` pierwsze $k$ wątków bloku dodaje te sumy częściowe przez `atomicAdd` do globalnych tablic `sums` i `count` w VRAM-ie. Na sam koniec lekki kernel finalizujący ($k \cdot d$ wątków) dzieli łączne sumy przez łączną liczbę punktów w każdym klastrze (zostawiając stary centroid, jeśli `count[c] == 0`) i kopiuje nowe współrzędne do `__constant__`.

3. **`gpu2` (Biblioteka NVIDIA Thrust — gotowe algorytmy równoległe bez sortowania):**
   * Wysokopoziomowa implementacja oparta na oficjalnej bibliotece **NVIDIA Thrust** (wbudowanej w CUDA), w której zamiast ręcznie zarządzać wątkami i pamięcią współdzieloną, korzystamy z gotowych operacji równoległych na całych tablicach:
     * **Przypisanie punktów i zliczenie zmian w jednym przejściu:** Zamiast tworzyć w pamięci tablicę z numerami punktów `0..N-1` (co zmarnowałoby dodatkowe setki megabajtów VRAM-u), używamy wirtualnego iteratora liczącego (`thrust::counting_iterator`), który generuje indeksy w locie. Wywołujemy na nim jedną operację `thrust::transform_reduce`: dla każdego punktu nasz funktor czyta współrzędne z tablicy SoA, porównuje je z centroidami w `__constant__`, wpisuje nowy numer klastra do `labels[idx]` i zwraca `1` (gdy klaster się zmienił) lub `0` (gdy został ten sam), a Thrust od razu w trakcie działania sumuje wszystkie zmiany.
     * **Wyliczenie nowych centroidów bez sortowania danych:** Zamiast sortować punkty według klastrów, zostawiamy wszystkie dane w ich pierwotnym, ciągłym układzie SoA i liczymy nowe średnie osobno dla każdego z $k$ klastrów. Najpierw za pomocą `thrust::count` sprawdzamy, ile punktów w tablicy `labels` ma przypisany dany klaster `c`. Następnie dla każdego wymiaru przechodzimy liniowo przez odpowiedni fragment tablicy `data` z prostym warunkiem (`thrust::inner_product`): jeśli punkt należy do klastra `c`, dodaj jego współrzędną do sumy; w przeciwnym razie dodaj zero. Na koniec dzielimy zebraną sumę przez liczbę punktów w klastrze (zostawiając stary centroid przy pustym klastrze) i kopiujemy nowe współrzędne do `__constant__`.
     * **Dlaczego rezygnujemy z sortowania:** Przy skali do $50\text{ mln}$ punktów i $d = 20$ główna tablica zajmuje ok. $8\text{ GB}$ pamięci. Klasyczne podejście z sortowaniem wymagałoby alokacji dodatkowych wielkich tablic na klucze, indeksy i pamięć roboczą (ryzyko błędu *Out Of Memory* na karcie), a po posortowaniu indeksów wątki musiałyby skakać po losowych miejscach w pamięci, tracąc koalescencję odczytów. Nasze podejście zużywa zero dodatkowej pamięci na kopie tablic i zawsze czyta dane po kolei.

### Optymalizacja w czasie kompilacji: metaprogramowanie szablonowe (`template <int D>`)

Ponieważ ze specyfikacji zadania wiemy, że liczba wymiarów $d$ zawsze mieści się w przedziale $[1, 20]$, możemy przenieść wiedzę o wymiarze z czasu wykonania programu (*runtime*) do czasu kompilacji (*compile-time*). Zamiast przekazywać `d` jako zwykłą zmienną do kernela, czynimy ją parametrem szablonu `template <int D>`, dzięki czemu kompilator generuje 20 wyspecjalizowanych wersji funkcji - osobno dla każdego wymiaru.

#### Dlaczego działa to szybciej na GPU?
* **Uniknięcie ucieczki danych z rejestrów (*Register Spilling*):** Sprzętowe rejestry procesora GPU nie mają adresów jak zwykła pamięć i nie potrafią być indeksowane dynamicznie zmienną w pętli (np. `punkt[dim]`). Gdy rozmiar lokalnej tablicy współrzędnych punktu zależy od zmiennej `d`, kompilator nie może umieścić jej w rejestrach i przenosi ją do pamieci lokalnej, która fizycznie leży w wolnym VRAM-ie. Dzięki stałej szablonu `D` oraz dyrektywie `#pragma unroll` kompilator całkowicie rozwija pętle po wymiarach, zamieniając dynamiczne indeksy na stałe (`punkt[0]`, `punkt[1]`, ...), co pozwala trzymać współrzędne w najszybszych rejestrach karty.
* **Eliminacja narzutu obsługi pętli:** Rozwinięcie pętli usuwa instrukcje inkrementacji licznika oraz sprawdzania warunku stopu przy każdym porównywaniu punktu z kolejnymi centroidami.

#### Rekurencja szablonów zamiast ręcznego bloku `switch`
Ponieważ dokładną wartość `d` poznajemy dopiero po wczytaniu nagłówka pliku wejściowego, do uruchomienia właściwej wersji kernela wykorzystujemy **rekurencję szablonów**. Kompilator sam generuje w czasie kompilacji łańcuch wywołań od `D = 20` w dół aż do specjalizacji bazowej dla `D = 0`, która zatrzymuje rekurencję:

```cpp
// Krok ogólny: obsłuż wymiar D albo przekaż niżej do D - 1
template <int D>
struct Launcher {
    static void run(int runtime_d, /* argumenty kernela */) {
        if (runtime_d == D) {
            mojKernel<D><<<bloki, watki>>>(/* argumenty */);
        } else {
            Launcher<D - 1>::run(runtime_d, /* argumenty */);
        }
    }
};

// Warunek stopu (specjalizacja dla D = 0): koniec odliczania
template <>
struct Launcher<0> {
    static void run(int, /* argumenty */) {} 
};
```

---

## 4. Interfejs programu i formaty danych

### Kompilacja
Projekt kompilowany jest bez ostrzeżeń za pomocą narzędzia `make`:
```bash
make
```

### Wywołanie programu
```bash
./KMeans data_format computation_method input_file output_file
```
* `data_format`: `txt` lub `bin`
* `computation_method`: `gpu1`, `gpu2` lub `cpu`
* `input_file`: ścieżka do pliku wejściowego
* `output_file`: ścieżka do pliku wynikowego

*(Wielkość liter ma znaczenie, a wszystkie parametry są obowiązkowe. W przypadku podania błędnej liczby lub niepoprawnej wartości argumentów program wypisuje instrukcję poprawnego wywołania i kończy działanie z kodem błędu).*

### Formaty plików
* **Wejście `txt`:** Pierwsza linia zawiera trzy liczby całkowite `N d k`. Kolejne $N$ linii zawiera po $d$ liczb zmiennoprzecinkowych (separator dziesiętny: kropka). Odczyt realizowany funkcją `fscanf`.
* **Wejście `bin`:** Nagłówek $12$ bajtów ($3 \times \text{int}$: `N`, `d`, `k`), a następnie $N \cdot d$ 8-bajtowych liczb typu `double` zapisanych punkt po punkcie. Odczyt realizowany funkcją `fread`.
* **Wyjście (zawsze `txt`):** Pierwsze $k$ linii zawiera współrzędne finalnych centroidów (po $d$ liczb typu `double` rozdzielonych spacjami, z kropką jako separatorem dziesiętnym), kolejne $N$ linii zawiera przypisany numer klastra (`0` do `k - 1`) dla każdego punktu. Zapis realizowany funkcją `fprintf`.

---

## 5. Kontrola błędów, pomiary wydajności i raportowanie

### Kontrola poprawności wykonania
* Każde wywołanie API CUDA (`cudaMalloc`, `cudaMemcpy`, `cudaMemcpyToSymbol`) oraz uruchomienie kernela (`cudaGetLastError()`) jest chronione makrem `CUDA_CHECK`, które w przypadku błędu wypisuje komunikat diagnostyczny, zwalnia zaalokowane zasoby i przerywa program.

### Raportowanie na `stdout` i pomiary czasu
Program wypisuje na standardowe wyjście (w czystym formacie tekstowym, z możliwością przekierowania do pliku):
* **Parametry uruchomienia:** informacje o wczytywanych danych (liczba punktów $N$, liczba wymiarów $d$, liczba klastrów $k$, format danych `txt`/`bin`) oraz wybrany algorytm obliczeniowy (`cpu`, `gpu1` lub `gpu2`).
* **Przebieg faz, postęp i czasy wykonania:** komunikaty o rozpoczęciu, postępie (w miarę możliwości) oraz zmierzonym czasie trwania każdej fazy programu:
  1. Wczytanie danych z dysku i transpozycja AoS $\rightarrow$ SoA
  2. Alokacja pamięci oraz transfer danych CPU $\rightarrow$ GPU
  3. Obliczenia właściwe (z wypisaniem w każdej iteracji jej numeru oraz liczby punktów, które zmieniły przypisanie do klastra)
  4. Transfer wyników GPU $\rightarrow$ CPU
  5. Zapis wyników do pliku tekstowego
