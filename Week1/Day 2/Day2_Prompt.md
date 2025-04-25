Awesome! I’ll design three space-themed Python challenges that strengthen your intermediate skills—especially focusing on concepts like object-oriented programming, data handling, and functional thinking. These will build your Python muscles in ways that’ll directly help when working with NumPy, pandas, and ML libraries later on.

I’ll be back shortly with your 2-hour challenge set!

# Space-Themed Python Challenges for Intermediate Learners

## 1. Mission Mars Rover Navigation

**Description:** You are the software engineer for a Mars rover mission, tasked with guiding the rover across the Martian plateau. The plateau is a grid defined by its width and height (in arbitrary units). The rover’s position is given by X-Y coordinates on this grid along with a facing direction (N, E, S, or W for north, east, south, west). The rover can receive a series of command letters:  
- **`L`** – turn 90° left (without moving from the current spot)  
- **`R`** – turn 90° right  
- **`M`** – move forward one grid unit in the direction it’s currently facing  

Write a program to simulate the rover’s movement given an initial position and a string of commands. The rover should not move outside the boundaries of the grid. After executing all commands, output the rover’s final coordinates and orientation.

**Example:** Suppose the grid size is 5x5, the rover starts at position **(1, 2)** facing North (`N`), and the command sequence is **`LMLMLMLMM`**.  
- Starting at (1,2,N). The rover turns left to face West, moves forward to (0,2), turns left to face South, moves to (0,1), turns left to East, moves to (1,1), turns left to North, then moves forward twice to (1,3).  
- **Final Output:** `1 3 N` (meaning the rover ends at coordinates x=1, y=3 facing North).  

**Suggested Extensions:**  
- Allow multiple rovers! Modify your program to handle several rovers on the same grid (each with its own start and command sequence), outputting each final position.  
- Introduce obstacles on the grid. For example, treat certain coordinates as boulders; if the rover is about to move into an obstacle, stop and report the obstacle encounter (ignore remaining commands).  
- Add more commands or features. You could support a **`B`** command for moving backward, or have the grid “wrap around” at the edges (as if the plateau is spherical). This encourages more complex condition checks in your code.  
- Refactor your solution with **OOP** principles: for instance, create a `Rover` class with methods like `turn_left()`, `turn_right()`, and `move()` to encapsulate the rover’s behavior. This makes your code easier to extend (e.g., adding different rover types or behaviors later on).

## 2. Exoplanet Data Analyzer (USED PANDAS INSTEAD)

**Description:** As a data scientist on the **Interstellar Survey Orbiter**, you’ve received a file containing data on newly discovered exoplanets. Each line in the file lists a planet’s name, its distance from Earth in light years, and its radius relative to Earth (e.g., 1.0 = same radius as Earth). Your task is to read this data file and generate some useful statistics for the astronomy team. Specifically, write a Python program that:  
- Reads the file line by line (you can assume a comma-separated format: `Name,Distance,Radius`).  
- Determines which planet is the farthest from Earth and how far it is.  
- Calculates the *average distance* of all listed planets.  

After computing these, the program should output the results in a clear format.

**Example:** If the data file contains the following lines:  
```
Planet,Distance_LY,Radius_Earths  
Proxima b,4.2,1.1  
Kepler-22b,600,2.4  
Zorgon-3,10.5,0.8  
```  
Your program would parse this information and could produce output like:  
```
Farthest planet: Kepler-22b (600 light years away)  
Average distance: 204.9 light years  
```  
*(In this example, Kepler-22b is the farthest, and the average of 4.2, 600, and 10.5 is 204.9).* 

**Suggested Extensions:**  
- Include additional analysis: for instance, count how many of the exoplanets are larger than Earth (radius > 1), or find the closest planet as well.  
- Sort the planets by distance or size and display the sorted list. This can deepen your use of Python’s list/tuple sorting and lambda functions.  
- Practice file **writing**: save the summary results (or even a sorted list of planets) to a new text file.  
- Use Python’s standard libraries to enhance your solution. For example, you could use the built-in `csv` module to handle parsing CSV files, or use the `statistics` module to easily compute the mean. You could even wrap planet data into a `Planet` class or use namedtuples for clarity.  
- For a more functional programming approach, try using list comprehensions or generator expressions to compute the distances list and averages, and use built-in functions like `max()` with a key or `sum()` for calculations. This will make your code concise and prepare you for using libraries like pandas later on.

## 3. Cosmic Cipher: Decode the Alien Message

**Description:** Earth’s deep-space network has just received a **mysterious encoded transmission** from an alien civilization. It’s your job to decode the message using Python! The aliens have used a simple cipher to obscure their message. In fact, it turns out to be a classic Caesar cipher – each letter of the original message is shifted by a fixed number of positions in the alphabet. (For example, with a shift of 5, `A` would be encoded as `F`, `B` as `G`, … `V` as `A`, `W` as `B`, etc., wrapping around the alphabet). Spaces and punctuation are left unchanged in the encoding.  

Write a program that takes the encoded alien message (you can store it as a string in your code or read from a file) and decodes it to reveal the plaintext. Assume you are given the shift value used by the cipher. For instance, if the shift is 5, your program needs to shift each letter 5 positions **backwards** to get the original message.

**Example:** Imagine the encoded message received is:  
```
BJ HTRJ NS UJFHJ
```  
If the known shift is 5, the program would decode this by shifting each letter 5 places back in the alphabet. The result would be:  
```
WE COME IN PEACE
```  
In this example, `BJ` shifts back to `WE`, `HTRJ` becomes `COME`, `NS` becomes `IN`, and `UJFHJ` becomes `PEACE`. The decoded output **“WE COME IN PEACE”** is the message in plain English.

**Suggested Extensions:**  
- **Brute-force decoding:** If you were **not** given the shift value, enhance your program to attempt all possible shifts (26 in English alphabets) and either display all results or automatically pick the one that forms a sensible English sentence. This is a common technique for cracking Caesar ciphers.  
- Make your decoder more robust by handling both uppercase and lowercase letters, or by preserving punctuation and numbers. (For example, the cipher might skip over non-letter characters without changing them.)  
- Turn your solution into a two-way utility: also allow the user to **encode** a message with a given shift (essentially the inverse operation). This could be designed as a command-line tool where the user specifies encode/decode and the shift value.  
- Use functional programming ideas to refactor the solution. For instance, use Python’s built-in functions or a mapping dictionary to map each character to its decoded form, or a list comprehension to apply the shift in one line. This practice will make transitioning to using libraries (which often apply functions to datasets element-wise, similar to map/filter mechanisms) more intuitive in the future.