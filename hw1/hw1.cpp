#include <iostream>
#include <cmath>
#include <iomanip>

void solver(float a, float b, float c) {
  if (a == 0 && b == 0 && c == 0) {
    std::cout << "any" << std::endl;
    return;
  }
  else if (a == 0 && b == 0) {
    std::cout << "incorrect" << std::endl;
    return;
  }
  else if (a == 0) {
    std::cout << -c / b << std::endl;
    return;
  }
  else {
    float D = b * b - 4 * a * c;

    if (D > 0) {
      float x_1 = (-b + sqrt(D)) / (2 * a);
      float x_2 = (-b - sqrt(D)) / (2 * a);
      std::cout << x_1 << ' ' << x_2 << std::endl;
      return;
    }
    else if (D == 0) {
      float x_1 = -b / (2 * a);
      std::cout << x_1 << std::endl;
      return;
    }
    else if (D < 0) {
      std::cout << "imaginary" << std::endl;
      return;
    }
  }
}

int main() {
  float a, b, c;
  std::cin >> a >> b >> c;
  std::cout << std::setprecision(6) << std::fixed;
  solver(a, b, c);
}
