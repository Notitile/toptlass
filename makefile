CXX = g++
CXXFLAGS = -Wall -std=c++17 -g

app: test.cpp
	$(CXX) $(CXXFLAGS) -o app test.cpp

clean:
	rm -f app