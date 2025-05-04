"""
Comprehensive LR Parser Test Script
Author: Claude AI
Date: May 2, 2025
"""

import os
import sys
import difflib
import argparse
import unittest
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from lr_parser import LRParser, Rule, TreeNode, Action, ActionType


class LRParserTests(unittest.TestCase):
    """Unit tests for the LR Parser functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.parser = LRParser("Grammar.txt", "ActionTable.txt", "GotoTable.txt", log_file="test_parser.log")
    
    def test_tokenization(self):
        """Test the tokenization function"""
        input_str = "id + id * id"
        expected = ["id", "+", "id", "*", "id", "$"]
        tokens = self.parser.tokenize(input_str)
        self.assertEqual(tokens, expected, "Tokenization failed")
    
    def test_grammar_loading(self):
        """Test grammar rule loading"""
        # Verify grammar has been loaded correctly
        self.assertTrue(len(self.parser.grammar) > 0, "Grammar rules not loaded")
        
        # Check if we have specific rules we expect
        # This assumes specific content in the grammar file
        has_start_rule = any(rule.left == "S'" for rule in self.parser.grammar)
        self.assertTrue(has_start_rule, "Grammar missing start rule")
    
    def test_action_table(self):
        """Test action table loading"""
        # Verify action table has been loaded correctly
        self.assertTrue(len(self.parser.action_table) > 0, "Action table not loaded")
        
        # Verify state 0 has valid actions
        self.assertTrue(0 in self.parser.action_table, "State 0 missing from action table")
        self.assertTrue(len(self.parser.action_table[0]) > 0, "No actions for state 0")
    
    def test_goto_table(self):
        """Test goto table loading"""
        # Verify goto table has been loaded correctly
        self.assertTrue(len(self.parser.goto_table) > 0, "Goto table not loaded")
        
        # Verify state 0 has valid gotos
        self.assertTrue(0 in self.parser.goto_table, "State 0 missing from goto table")
    
    def test_successful_parse(self):
        """Test a simple valid input"""
        input_str = "id + id"
        success, _ = self.parser.parse(input_str)
        self.assertTrue(success, f"Failed to parse valid input: '{input_str}'")
    
    def test_invalid_parse(self):
        """Test an invalid input"""
        input_str = "id id +"  # Invalid syntax
        success, _ = self.parser.parse(input_str)
        self.assertFalse(success, f"Incorrectly parsed invalid input: '{input_str}'")
    
    def test_parse_tree_construction(self):
        """Test that the parse tree is constructed correctly"""
        input_str = "id + id"
        self.parser.parse(input_str)
        
        # Verify that we have a parse tree
        self.assertIsNotNone(self.parser.root, "No parse tree constructed")
        
        # Verify the root node has the expected value (usually S' or similar)
        self.assertIn(self.parser.root.value, self.parser.non_terminals, 
                     f"Root value {self.parser.root.value} is not a non-terminal")
        
        # Verify the tree has some depth
        self.assertTrue(len(self.parser.root.children) > 0, "Parse tree has no depth")


def run_test_suite():
    """Run the unit test suite"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


def verify_files_exist(files: List[str]) -> bool:
    """Verify that required files exist"""
    missing_files = []
    for file in files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True


def compare_files(file1: str, file2: str, show_diff: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Compare two files and return if they match.
    Optionally return a diff of the differences.
    """
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            content1 = f1.readlines()
            content2 = f2.readlines()
            
            if content1 == content2:
                return True, None
            
            if show_diff:
                diff = difflib.unified_diff(content1, content2, fromfile=file1, tofile=file2)
                return False, ''.join(diff)
            else:
                return False, None
    except Exception as e:
        return False, f"Error comparing files: {str(e)}"


def test_all_inputs(output_prefix: str = "my_output", expected_prefix: str = "output", 
                    show_diffs: bool = True, stop_on_error: bool = False,
                    test_range: List[int] = None):
    """
    Test the parser against all input files and compare results with expected outputs.
    
    Args:
        output_prefix: Prefix for generated output files
        expected_prefix: Prefix for expected output files
        show_diffs: Whether to show diffs for mismatches
        stop_on_error: Whether to stop testing after the first error
        test_range: List of test numbers to run (if None, run all available tests)
    """
    # Create parser
    parser = LRParser("Grammar.txt", "ActionTable.txt", "GotoTable.txt", log_file="batch_test_parser.log")
    
    # Get available input files
    input_files = [f for f in os.listdir() if f.startswith("input") and f.endswith(".txt")]
    input_numbers = [int(f.replace("input", "").replace(".txt", "")) for f in input_files]
    input_numbers.sort()
    
    if test_range:
        input_numbers = [n for n in input_numbers if n in test_range]
    
    # Statistics
    total_tests = len(input_numbers)
    passed_tests = 0
    failed_tests = 0
    
    print(f"\n===== Running LR Parser Tests on {total_tests} input files =====\n")
    
    for test_num in input_numbers:
        input_file = f"input{test_num}.txt"
        output_file = f"{output_prefix}{test_num}.txt"
        expected_output_file = f"{expected_prefix}{test_num}.txt"
        
        print(f"Test #{test_num}: Processing {input_file}...")
        
        # Check if input file exists (should always be true, but double-check)
        if not os.path.exists(input_file):
            print(f"  ❌ Error: {input_file} not found.")
            failed_tests += 1
            if stop_on_error:
                break
            continue
        
        try:
            # Read input
            with open(input_file, 'r') as file:
                input_str = file.read().strip()
                print(f"  Input: '{input_str}'")
            
            # Parse input
            success, _ = parser.parse(input_str)
            
            # Save output
            parser.save_output(output_file)
            
            status = "✅ Parsing successful" if success else "❌ Parsing failed"
            print(f"  {status}")
            print(f"  Output saved to {output_file}")
            
            # Compare with expected output if available
            if os.path.exists(expected_output_file):
                print(f"  Comparing with {expected_output_file}...")
                match, diff = compare_files(output_file, expected_output_file, show_diff=show_diffs)
                
                if match:
                    print("  ✅ Output exactly matches expected output!")
                    passed_tests += 1
                else:
                    print("  ❌ Output differs from expected output.")
                    if show_diffs and diff:
                        print("\n  --- Differences ---")
                        print(diff)
                        print("  ------------------\n")
                    failed_tests += 1
                    if stop_on_error:
                        break
            else:
                print(f"  ⚠️ Warning: Expected output file {expected_output_file} not found. Cannot verify correctness.")
                # Not counting as passed or failed since we can't verify
        
        except Exception as e:
            print(f"  ❌ Error during test execution: {str(e)}")
            failed_tests += 1
            if stop_on_error:
                break
        
        print()
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"Total tests:  {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    return passed_tests == total_tests


def test_custom_input(input_string: str, output_file: str = "custom_output.txt"):
    """
    Test the parser with a custom input string.
    
    Args:
        input_string: The input string to parse
        output_file: Where to save the output
    """
    parser = LRParser("Grammar.txt", "ActionTable.txt", "GotoTable.txt", log_file="custom_test_parser.log")
    
    print(f"\n===== Testing Custom Input =====")
    print(f"Input: '{input_string}'")
    
    try:
        # Parse input
        success, _ = parser.parse(input_string)
        
        # Save output
        parser.save_output(output_file)
        
        status = "✅ Parsing successful" if success else "❌ Parsing failed"
        print(f"{status}")
        print(f"Output saved to {output_file}")
        
        # Display parse tree if parsing was successful
        if success:
            print("\nParse Tree:")
            print(parser.print_parse_tree())
        
        return success
    
    except Exception as e:
        print(f"❌ Error during parsing: {str(e)}")
        return False


def main():
    """Main function for the test script"""
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Test LR Parser Implementation')
    parser.add_argument('--unit-tests', action='store_true', 
                        help='Run unit tests')
    parser.add_argument('--batch-tests', action='store_true',
                        help='Run batch tests on input files')
    parser.add_argument('--test-numbers', type=int, nargs='+',
                        help='Specific test numbers to run (e.g., --test-numbers 1 3 5)')
    parser.add_argument('--output-prefix', type=str, default='my_output',
                        help='Prefix for output files')
    parser.add_argument('--compare-with', type=str, default='output',
                        help='Prefix for expected output files to compare with')
    parser.add_argument('--no-diff', action='store_true',
                        help='Do not show diffs for mismatches')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop testing after the first error')
    parser.add_argument('--custom-input', type=str,
                        help='Parse a custom input string')
    parser.add_argument('--custom-output', type=str, default='custom_output.txt',
                        help='Output file for custom input parsing')
    
    args = parser.parse_args()
    
    # If no test type is specified, run both by default
    if not (args.unit_tests or args.batch_tests or args.custom_input):
        args.unit_tests = True
        args.batch_tests = True
    
    # Check if required files exist
    required_files = ["Grammar.txt", "ActionTable.txt", "GotoTable.txt"]
    if not verify_files_exist(required_files):
        print("Exiting due to missing required files.")
        return 1
    
    # Run tests based on arguments
    all_passed = True
    
    if args.unit_tests:
        print("\n===== Running Unit Tests =====")
        run_test_suite()
    
    if args.batch_tests:
        batch_success = test_all_inputs(
            output_prefix=args.output_prefix,
            expected_prefix=args.compare_with,
            show_diffs=not args.no_diff,
            stop_on_error=args.stop_on_error,
            test_range=args.test_numbers
        )
        all_passed = all_passed and batch_success
    
    if args.custom_input:
        custom_success = test_custom_input(args.custom_input, args.custom_output)
        all_passed = all_passed and custom_success
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())