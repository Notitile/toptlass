CXX = g++
CXXFLAGS = -Wall -std=c++17 -g

app: main.cpp
	$(CXX) $(CXXFLAGS) -o app main.cpp

clean:
	rm -f app