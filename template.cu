#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================================================
// 1. MAKRO DO OBSŁUGI BŁĘDÓW CUDA
// ============================================================================
// Funkcje CUDA zwracają typ cudaError_t. To makro automatycznie sprawdza,
// czy operacja (np. alokacja pamięci lub kopiowanie) zakończyła się sukcesem.
// Jeśli wystąpi błąd, wypisze plik, numer linii oraz opis błędu i zakończy program.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "Błąd CUDA w pliku %s (linia %d): %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================================
// 2. KERNEL (FUNKCJA WYKONYWANA NA KARCIE GRAFICZNEJ - DEVICE)
// ============================================================================
// __global__ oznacza, że funkcja jest wywoływana z CPU (Host), ale wykonuje się
// na GPU (Device). Musi zawsze zwracać typ void.
__global__ void myKernel(int *d_data, int n) {
    // Obliczenie globalnego, unikalnego indeksu wątku:
    // blockIdx.x  - indeks aktualnego bloku w siatce (Grid)
    // blockDim.x  - liczba wątków w jednym bloku
    // threadIdx.x - indeks wątku wewnątrz aktualnego bloku
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Zabezpieczenie przed wyjściem poza zakres tablicy (ważne, gdy liczba
    // wszystkich uruchomionych wątków jest większa niż rozmiar danych 'n')
    if (idx < n) {
        // TUTAJ WPISZ LOGIKĘ SWOJEGO KERNELA
        // Przykład: pomnożenie każdego elementu przez 2
        d_data[idx] = d_data[idx] * 2;
    }
}

// ============================================================================
// 3. FUNKCJA GŁÓWNA (WYKONYWANA NA PROCESORZE - HOST)
// ============================================================================
int main() {
    // --- A. Konfiguracja rozmiaru danych ---
    const int N = 10; // Liczba elementów
    const size_t bytes = N * sizeof(int); // Rozmiar w bajtach

    // --- B. Alokacja pamięci na CPU (Host) ---
    // Dobrą praktyką jest dodawanie prefiksu 'h_' dla zmiennych na CPU (Host)
    int *h_data = (int*)malloc(bytes);
    if (h_data == NULL) {
        fprintf(stderr, "Błąd alokacji pamięci na CPU!\n");
        return EXIT_FAILURE;
    }

    // Inicjalizacja danych początkowych na CPU
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    // --- C. Alokacja pamięci na GPU (Device) ---
    // Dobrą praktyką jest dodawanie prefiksu 'd_' dla zmiennych na GPU (Device)
    int *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // --- D. Kopiowanie danych z CPU (Host) na GPU (Device) ---
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // --- E. Konfiguracja i uruchomienie kernela ---
    // Ustalamy ile wątków ma być w jednym bloku (zazwyczaj wielokrotność 32, np. 128, 256, 512)
    int threadsPerBlock = 256;
    // Obliczamy ile bloków potrzebujemy, aby pokryć wszystkie N elementów (zaokrąglenie w górę)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Uruchomienie kernela na GPU za pomocą składni <<<liczba_bloków, wątki_na_blok>>>
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Sprawdzenie, czy samo uruchomienie kernela nie wywołało błędu (np. błędna konfiguracja wątków)
    CUDA_CHECK(cudaGetLastError());

    // Oczekiwanie na zakończenie pracy GPU (wywołania kerneli są asynchroniczne!)
    // Zwróci też ewentualne błędy, które wystąpiły w trakcie działania samego kernela.
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- F. Kopiowanie wyników z powrotem z GPU (Device) na CPU (Host) ---
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // --- G. Weryfikacja / Wypisanie wyników ---
    printf("Wyniki po obliczeniach na GPU:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // --- H. Zwolnienie pamięci (Sprzątanie) ---
    CUDA_CHECK(cudaFree(d_data)); // Zwolnienie pamięci na GPU
    free(h_data);                 // Zwolnienie pamięci na CPU

    // Opcjonalnie: reset urządzenia na koniec programu (pomocne przy profilowaniu)
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
