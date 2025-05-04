"""
LR Parser Implementation for SWE 204 Homework-1
Author: Claude AI
"""

import os
import sys
import logging
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime


class ActionType(Enum):
    """Enum for action types in LR parser"""
    SHIFT = "s"
    REDUCE = "r"
    ACCEPT = "a"
    ERROR = "e"


class Action:
    """Represents an LR parser action"""
    
    def __init__(self, action_type: ActionType, value: int = None):
        self.action_type = action_type
        self.value = value
    
    def __str__(self):
        if self.action_type == ActionType.ACCEPT:
            return "accept"
        elif self.action_type == ActionType.ERROR:
            return "error"
        else:
            return f"{self.action_type.value}{self.value}"
    
    @staticmethod
    def parse(action_str: str) -> 'Action':
        """Parse action string from the action table"""
        if not action_str or action_str == " ":
            return Action(ActionType.ERROR)
        
        if action_str == "acc":
            return Action(ActionType.ACCEPT)
        
        action_type = ActionType(action_str[0])
        value = int(action_str[1:]) if len(action_str) > 1 else None
        
        return Action(action_type, value)


class Rule:
    """Represents a grammar rule"""
    
    def __init__(self, left: str, right: List[str]):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"{self.left} -> {' '.join(self.right)}"


class TreeNode:
    """Represents a node in the parse tree"""
    
    def __init__(self, value: str, children: List['TreeNode'] = None):
        self.value = value
        self.children = children or []
    
    def add_child(self, node: 'TreeNode'):
        self.children.append(node)
    
    def __str__(self):
        return self.value


def setup_logger(log_file="parser.log", console_level=logging.INFO, file_level=logging.DEBUG, name="LRParser"):
    """Setup logger for the application"""
    # Eğer bu isimde bir logger zaten varsa, onu döndür (çift kayıtları önle)
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class LRParser:
    """LR parser for basic arithmetic expressions"""
    
    def __init__(self, grammar_file: str, action_table_file: str, goto_table_file: str, log_file="parser.log"):
        # Setup logger
        self.logger = setup_logger(log_file)
        self.logger.info("Initializing LR Parser")
        
        self.logger.debug(f"Loading grammar from {grammar_file}")
        self.grammar = self._load_grammar(grammar_file)
        
        self.logger.debug(f"Loading action table from {action_table_file}")
        self.action_table = self._load_action_table(action_table_file)
        
        self.logger.debug(f"Loading goto table from {goto_table_file}")
        self.goto_table = self._load_goto_table(goto_table_file)
        
        self.logger.debug("Extracting non-terminals and terminals")
        self.non_terminals = self._get_non_terminals()
        self.terminals = self._get_terminals()
        
        self.trace_table = []
        self.root = None  # Root of the parse tree
        
        self.logger.info("LR Parser initialized successfully")
        self.logger.debug(f"Grammar loaded: {len(self.grammar)} rules")
        self.logger.debug(f"Action table loaded: {len(self.action_table)} states")
        self.logger.debug(f"Goto table loaded: {len(self.goto_table)} states")
        self.logger.debug(f"Non-terminals: {self.non_terminals}")
        self.logger.debug(f"Terminals: {self.terminals}")
    
    def _load_grammar(self, filename: str) -> List[Rule]:
        """Load grammar rules from file"""
        rules = []
        
        try:
            with open(filename, 'r') as file:
                self.logger.debug(f"Reading grammar file: {filename}")
                for i, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split("->")
                    if len(parts) != 2:
                        self.logger.warning(f"Invalid grammar rule format at line {i}: {line}")
                        continue
                    
                    left = parts[0].strip()
                    right = parts[1].strip().split()
                    rule = Rule(left, right)
                    rules.append(rule)
                    self.logger.debug(f"Loaded rule: {rule}")
            
            self.logger.info(f"Successfully loaded {len(rules)} grammar rules")
            return rules
            
        except FileNotFoundError:
            self.logger.error(f"Grammar file not found: {filename}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error loading grammar file: {e}")
            sys.exit(1)
    
    def _load_action_table(self, filename: str) -> Dict[int, Dict[str, Action]]:
        """Load action table from file"""
        action_table = {}
        
        try:
            with open(filename, 'r') as file:
                self.logger.debug(f"Reading action table file: {filename}")
                
                # Read header line to get terminals
                header = file.readline().strip().split()
                terminals = header[1:]  # First column is state
                self.logger.debug(f"Action table terminals: {terminals}")
                
                for line_num, line in enumerate(file, 2):  # Start from 2 because header is line 1
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    try:
                        state = int(parts[0])
                        action_table[state] = {}
                        
                        for i, term in enumerate(terminals):
                            if i + 1 < len(parts) and parts[i + 1]:
                                action = Action.parse(parts[i + 1])
                                action_table[state][term] = action
                                self.logger.debug(f"Loaded action[{state},{term}] = {action}")
                    except ValueError:
                        self.logger.warning(f"Invalid state number at line {line_num}: {parts[0]}")
                    except Exception as e:
                        self.logger.warning(f"Error parsing action table at line {line_num}: {e}")
            
            self.logger.info(f"Successfully loaded action table with {len(action_table)} states")
            return action_table
            
        except FileNotFoundError:
            self.logger.error(f"Action table file not found: {filename}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error loading action table file: {e}")
            sys.exit(1)
    
    def _load_goto_table(self, filename: str) -> Dict[int, Dict[str, int]]:
        """Load goto table from file"""
        goto_table = {}
        
        try:
            with open(filename, 'r') as file:
                self.logger.debug(f"Reading goto table file: {filename}")
                
                # Read header line to get non-terminals
                header = file.readline().strip().split()
                non_terminals = header[1:]  # First column is state
                self.logger.debug(f"Goto table non-terminals: {non_terminals}")
                
                for line_num, line in enumerate(file, 2):  # Start from 2 because header is line 1
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    try:
                        state = int(parts[0])
                        goto_table[state] = {}
                        
                        for i, non_term in enumerate(non_terminals):
                            if i + 1 < len(parts) and parts[i + 1].isdigit():
                                goto = int(parts[i + 1])
                                goto_table[state][non_term] = goto
                                self.logger.debug(f"Loaded goto[{state},{non_term}] = {goto}")
                    except ValueError:
                        self.logger.warning(f"Invalid state number at line {line_num}: {parts[0]}")
                    except Exception as e:
                        self.logger.warning(f"Error parsing goto table at line {line_num}: {e}")
            
            self.logger.info(f"Successfully loaded goto table with {len(goto_table)} states")
            return goto_table
            
        except FileNotFoundError:
            self.logger.error(f"Goto table file not found: {filename}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error loading goto table file: {e}")
            sys.exit(1)
    
    def _get_non_terminals(self) -> Set[str]:
        """Get all non-terminals from the grammar"""
        return set(rule.left for rule in self.grammar)
    
    def _get_terminals(self) -> Set[str]:
        """Get all terminals from the grammar"""
        terminals = set()
        for rule in self.grammar:
            for symbol in rule.right:
                if symbol not in self.non_terminals:
                    terminals.add(symbol)
        return terminals
    
    def tokenize(self, input_str: str) -> List[str]:
        """Convert input string to a list of tokens"""
        # Just splitting by spaces for this simple implementation
        # In a more complex parser, we would have a proper lexer
        tokens = input_str.strip().split()
        tokens.append("$")  # Add end marker
        return tokens
    
    def parse(self, input_str: str) -> Tuple[bool, List[Dict]]:
        """
        Parse the input string using LR parsing technique.
        Returns a tuple of (success, trace_table)
        """
        self.logger.info(f"Starting to parse input: '{input_str}'")
        
        tokens = self.tokenize(input_str)
        self.logger.debug(f"Tokenized input: {tokens}")
        
        stack = [0]  # State stack initialized with state 0
        symbol_stack = []  # Symbol stack
        node_stack = []  # Node stack for parse tree construction
        self.trace_table = []  # Reset trace table
        
        token_idx = 0
        current_token = tokens[token_idx]
        
        self.logger.info("Starting LR parsing algorithm")
        step_count = 0
        
        while True:
            step_count += 1
            current_state = stack[-1]
            
            self.logger.debug(f"Step {step_count}: State={current_state}, Current token={current_token}")
            self.logger.debug(f"Stack: {stack}")
            self.logger.debug(f"Symbol stack: {symbol_stack}")
            self.logger.debug(f"Remaining input: {tokens[token_idx:]}")
            
            # Record the current state for trace table
            trace_entry = {
                "stack": " ".join(map(str, stack)),
                "symbols": " ".join(symbol_stack),
                "input": " ".join(tokens[token_idx:]),
                "action": ""
            }
            
            # Check if the current token is valid
            if current_token not in self.action_table[current_state]:
                error_msg = f"Error: Invalid token '{current_token}' at position {token_idx}"
                self.logger.error(error_msg)
                trace_entry["action"] = "error: invalid token"
                self.trace_table.append(trace_entry)
                return False, self.trace_table
            
            action = self.action_table[current_state].get(current_token)
            
            if not action or action.action_type == ActionType.ERROR:
                # Error state
                error_msg = f"Error: No valid action for state {current_state} and token '{current_token}'"
                self.logger.error(error_msg)
                trace_entry["action"] = "error"
                self.trace_table.append(trace_entry)
                return False, self.trace_table
            
            trace_entry["action"] = str(action)
            self.logger.debug(f"Action: {action}")
            
            if action.action_type == ActionType.SHIFT:
                # Shift operation
                self.logger.debug(f"Performing SHIFT to state {action.value}")
                stack.append(action.value)  # Push next state
                symbol_stack.append(current_token)  # Push current token
                
                # Create leaf node for terminal
                node = TreeNode(current_token)
                node_stack.append(node)
                
                token_idx += 1
                if token_idx < len(tokens):
                    current_token = tokens[token_idx]
                
            elif action.action_type == ActionType.REDUCE:
                # Reduce operation
                rule = self.grammar[action.value - 1]  # Rule index is 1-based
                self.logger.debug(f"Performing REDUCE with rule {action.value}: {rule}")
                
                # Pop states and symbols according to the rule length
                rule_len = len(rule.right)
                
                # Create tree node for non-terminal
                node = TreeNode(rule.left)
                
                # Pop children for current reduction
                children = []
                for _ in range(rule_len):
                    if node_stack:
                        children.insert(0, node_stack.pop())
                
                # Add children to the current node
                node.children = children
                
                # Pop states and symbols
                for _ in range(rule_len):
                    stack.pop()
                    if symbol_stack:
                        symbol_stack.pop()
                
                # Get new state from goto table
                current_state = stack[-1]
                next_state = self.goto_table[current_state].get(rule.left, None)
                
                if next_state is None:
                    error_msg = f"Error: No goto state for state {current_state} and non-terminal {rule.left}"
                    self.logger.error(error_msg)
                    trace_entry["action"] = "error: no goto state"
                    self.trace_table.append(trace_entry)
                    return False, self.trace_table
                
                self.logger.debug(f"Got GOTO state {next_state}")
                
                # Push new state and symbol
                stack.append(next_state)
                symbol_stack.append(rule.left)
                node_stack.append(node)
                
            elif action.action_type == ActionType.ACCEPT:
                # Accept state
                self.logger.info("Input accepted!")
                self.trace_table.append(trace_entry)
                if node_stack:
                    self.root = node_stack[0]  # Set parse tree root
                    self.logger.debug("Parse tree constructed successfully")
                return True, self.trace_table
            
            else:
                # Unknown action
                error_msg = f"Error: Unknown action type {action.action_type}"
                self.logger.error(error_msg)
                trace_entry["action"] = "error: unknown action"
                self.trace_table.append(trace_entry)
                return False, self.trace_table
            
            self.trace_table.append(trace_entry)
    
    def print_parse_tree(self, node: TreeNode = None, level: int = 0) -> str:
        """Print the parse tree recursively"""
        if node is None:
            node = self.root
            if node is None:
                return "No parse tree available"
        
        result = ""
        indent = "  " * level
        result += f"{indent}{node.value}\n"
        
        for child in node.children:
            result += self.print_parse_tree(child, level + 1)
        
        return result
    
    def print_trace_table(self) -> str:
        """Print the trace table"""
        result = "Stack | Input | Action\n"
        result += "-" * 50 + "\n"
        
        for entry in self.trace_table:
            stack_symbols = entry["stack"]
            if "symbols" in entry:
                stack_symbols += " " + entry["symbols"]
            
            result += f"{stack_symbols} | {entry['input']} | {entry['action']}\n"
        
        return result
    
    def save_output(self, filename: str):
        """Save the trace table and parse tree to a file"""
        self.logger.info(f"Saving output to file: {filename}")
        
        try:
            with open(filename, 'w') as file:
                # Write trace table
                file.write("PARSE TRACING TABLE:\n")
                file.write("Stack | Input | Action\n")
                file.write("-" * 50 + "\n")
                
                for entry in self.trace_table:
                    stack_str = entry["stack"]
                    if "symbols" in entry:
                        stack_str += " " + entry["symbols"]
                    
                    file.write(f"{stack_str} | {entry['input']} | {entry['action']}\n")
                
                # Write parse tree
                file.write("\nPARSE TREE:\n")
                if self.root:
                    parse_tree = self.print_parse_tree()
                    file.write(parse_tree)
                    self.logger.debug("Parse tree written to output file")
                else:
                    file.write("No parse tree available - parsing failed")
                    self.logger.warning("No parse tree available - parsing failed")
            
            self.logger.info(f"Output successfully saved to {filename}")
        
        except Exception as e:
            self.logger.error(f"Error saving output to file {filename}: {e}")
            print(f"Error: Could not save output to {filename}")
            raise


def main():
    """Main function to run the LR parser"""
    # Setup main logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"lr_parser_{timestamp}.log"
    main_logger = setup_logger(log_file)
    main_logger.info("LR Parser started")
    
    if len(sys.argv) < 2:
        main_logger.error("No input file specified")
        print("Usage: python lr_parser.py <input_file>")
        return
    
    try:
        input_file = sys.argv[1]
        output_file = input_file.replace("input", "output")
        grammar_file = "Grammar.txt"
        action_table_file = "ActionTable.txt"
        goto_table_file = "GotoTable.txt"
        
        main_logger.info(f"Input file: {input_file}")
        main_logger.info(f"Output file: {output_file}")
        main_logger.info(f"Grammar file: {grammar_file}")
        main_logger.info(f"Action table file: {action_table_file}")
        main_logger.info(f"Goto table file: {goto_table_file}")
        
        # Aynı log dosyasını parser için de kullan
        parser = LRParser(grammar_file, action_table_file, goto_table_file, log_file)
        
        # Check if input file exists
        if not os.path.exists(input_file):
            main_logger.error(f"Input file not found: {input_file}")
            print(f"Error: Input file not found: {input_file}")
            return
        
        # Read input
        try:
            with open(input_file, 'r') as file:
                input_str = file.read().strip()
                main_logger.info(f"Read input from {input_file}: '{input_str}'")
        except Exception as e:
            main_logger.error(f"Error reading input file: {e}")
            print(f"Error reading input file: {e}")
            return
        
        # Parse input
        try:
            main_logger.info("Starting parsing process")
            success, _ = parser.parse(input_str)
            main_logger.info(f"Parsing {'successful' if success else 'failed'}")
        except Exception as e:
            main_logger.error(f"Error during parsing: {e}")
            print(f"Error during parsing: {e}")
            return
        
        # Save output
        try:
            parser.save_output(output_file)
            main_logger.info(f"Output saved to {output_file}")
        except Exception as e:
            main_logger.error(f"Error saving output: {e}")
            print(f"Error saving output: {e}")
            return
        
        print(f"Parsing {'successful' if success else 'failed'}.")
        print(f"Output saved to {output_file}")
        print(f"Logs saved to {log_file}")
        
        main_logger.info("LR Parser completed successfully")
    
    except Exception as e:
        main_logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()