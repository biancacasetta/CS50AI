import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells
        else:
            return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0 and self.cells:
            return self.cells
        else:
            return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        self.moves_made.add(cell)
        self.mark_safe(cell)
        kb_updated = True

        # based on given cell, calculate neighbouring cells
        # check that they exist on board (no inexistent indexes)
        # check if any of the cells have been clicked
        # filter those cells
        # create Sentence with remaining cells and count

        neighboring_cells = set()
        row, column = cell
        for r in range(row-1, row+2):
            for c in range(column-1, column+2):
                if (r, c) != cell and 0 <= r < self.height and 0 <= c < self.width:
                    neighbour_cell = (r, c)
                    neighboring_cells.add(neighbour_cell)                

        invalid_cells = set()                
        for nc in neighboring_cells:
            if nc in self.mines:
                invalid_cells.add(nc)
                count -= 1
            if nc in self.safes:
                invalid_cells.add(nc)
        
        neighboring_cells -= invalid_cells
        new_sentence = Sentence(neighboring_cells, count)
        
        if new_sentence not in self.knowledge:
            self.knowledge.append(new_sentence)

        # check for possible conclusions based on new cell info
        # if conclusions are found (known mines/safes), update knowledge base
        # after update, check if new conclusions are possible and if so update again

        while kb_updated:
            kb_updated = False
            known_mines = set()
            known_safes = set()

            for sentence in self.knowledge:
                if sentence.known_mines():
                    kb_updated = True
                    known_mines = known_mines.union(sentence.known_mines())
                elif sentence.known_safes():
                    kb_updated = True
                    known_safes = known_safes.union(sentence.known_safes())

                discarded = set()
                for c in sentence.cells:
                    if c in self.mines:
                        kb_updated = True
                        discarded.add(c)
                
                sentence.cells -= discarded

            for mine in known_mines:
                self.mark_mine(mine)
            
            for safe in known_safes:
                self.mark_safe(safe)

            self.knowledge = [s for s in self.knowledge if s.cells]

            inferences = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1.cells != s2.cells and s2.cells.issubset(s1.cells):
                        inference = Sentence(s1.cells - s2.cells, s1.count - s2.count)
                        if inference not in self.knowledge and inference not in inferences:
                            kb_updated = True
                            inferences.append(inference)
            
            self.knowledge.extend(inferences)


    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        safes = self.safes - self.moves_made
        if safes:
            return safes.pop()
        
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        available_cells = set()

        for r in range(self.width):
            for c in range(self.height):
                available_cells.add((r, c))
        
        available_cells -= self.moves_made
        available_cells -= self.mines

        if available_cells:
            return available_cells.pop()
        else:
            return None