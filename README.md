# chess218

![image](https://github.com/Tobs40/chess218/assets/63099057/30f000d3-0ad6-45cf-92dd-059dc5b34ee6)

A code snippet that, using the solver Gurobi, proves, that there is no legal chess position with more than 218 moves.
Can be configured to solve very similar problems (144 moves without promotions, with certain restrictions on piece counts, ...).
Does not check whether the position can be reached from the initial position. There are tools for that: https://github.com/peterosterlund2/texel

Usage:
- install gurobipy via pip
- install Gurobi on your system
- obtain and activate an academic license from https://www.gurobi.com/
- make sure gurobipy finds Gurobi, this is usually the case
- modify the script according to your needs
- start running it and wait for the results

Solutions (Correct, illegal, wrong) will be written to a folder as images as well as text files with the corresponding fen.
At the end of the program, after the popup window with the optimal solution is closed, a text file with the FENs of all correct solutions is created.




Check out the corresponding Lichess article: https://lichess.org/@/Tobs40/blog
