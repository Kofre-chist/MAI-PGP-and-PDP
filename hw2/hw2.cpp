#include <iostream>
#include <iomanip>

void bubble_sort(float *vec, int n) {
  float tmp = 0;
  for (int i = 0; i < n - 1; ++i) {
    for (int j = 0; j < n - 1; ++j) {
      if (vec[j] > vec[j + 1]) {
        tmp = vec[j];
        vec[j] = vec[j + 1];
        vec[j + 1] = tmp;
      }
    }
  }
}

int main() {
  int n;
  std::cin >> n;
  float* vec = (float*)malloc(sizeof(float) * n);
  for (int i = 0; i < n; ++i) {
    std::cin >> vec[i];
  }
  bubble_sort(vec, n);
  std::cout << std::setprecision(6) << std::scientific;
  for (int i = 0; i < n; ++i) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;

  free(vec);

  return 0;
}
