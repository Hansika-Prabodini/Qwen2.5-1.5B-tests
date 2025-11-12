#!/usr/bin/env python3
"""
Data Generation Script for Derivatives Query Dataset

This script generates synthetic query-filter pairs for training a model to parse
natural language queries about derivatives trading data into structured JSON filters.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from tqdm import tqdm


class DerivativesDataGenerator:
    """Generator for derivatives query-filter pairs."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the generator by loading the CSV schema.
        
        Args:
            csv_path: Path to Derivatives_Lingo.csv
        """
        self.df = pd.read_csv(csv_path)
        self._build_schema_mappings()
        self._define_templates()
        
    def _build_schema_mappings(self):
        """Build mappings from the CSV for quick lookups."""
        # Group by table and column to get available fields
        self.tables = self.df['db_table'].unique().tolist()
        self.columns_by_table = {
            table: self.df[self.df['db_table'] == table]['db_column'].unique().tolist()
            for table in self.tables
        }
        
        # Map columns to their types
        self.column_types = {}
        for _, row in self.df.iterrows():
            key = (row['db_table'], row['db_column'])
            if key not in self.column_types:
                self.column_types[key] = row['db_type']
        
        # Group aliases and real names by table and column
        self.aliases_by_column = {}
        for _, row in self.df.iterrows():
            key = (row['db_table'], row['db_column'])
            if key not in self.aliases_by_column:
                self.aliases_by_column[key] = []
            self.aliases_by_column[key].append({
                'alias': row['alias'],
                'real_name': row['real_name'],
                'category': row['category']
            })
    
    def _define_templates(self):
        """Define query templates for different complexity levels."""
        # Simple templates (1 filter)
        self.simple_templates = [
            "show me {alias} trades",
            "get all {alias} positions",
            "find {real_name} transactions",
            "list {alias} entries",
            "display {real_name} records",
        ]
        
        # Medium templates (2-3 filters)
        self.medium_templates = [
            "show me {alias} trades over {value}",
            "get {real_name} positions with {column} greater than {value}",
            "find {alias} transactions between {date1} and {date2}",
            "list {alias} entries where {column} equals {value}",
            "show me {real_name} trades with {column} less than {value}",
            "get all {alias} positions where {column} is {value} and {column2} is {value2}",
        ]
        
        # Complex templates (4+ filters)
        self.complex_templates = [
            "show me {alias} trades over {value} with {column} between {val1} and {val2}",
            "find {real_name} positions where {column} is {value}, {column2} greater than {val2}, and {column3} equals {val3}",
            "get {alias} transactions with {column} over {value}, {column2} under {val2}, and dated after {date}",
        ]
        
        # Operators by data type
        self.operators_by_type = {
            'numeric': ['=', '!=', '>', '<', '>=', '<='],
            'bigint': ['=', '!=', '>', '<', '>=', '<='],
            'integer': ['=', '!=', '>', '<', '>=', '<='],
            'text': ['=', '!=', 'LIKE', 'IN'],
            'date': ['=', '!=', '>', '<', '>=', '<='],
            'timestamp': ['=', '!=', '>', '<', '>=', '<='],
        }
    
    def _get_random_value_for_type(self, db_type: str, column_name: str) -> Any:
        """Generate a random realistic value based on column type and name."""
        if db_type in ['numeric', 'bigint', 'integer']:
            # Different ranges based on column name
            if 'notional' in column_name.lower():
                return random.choice([100000, 500000, 1000000, 5000000, 10000000])
            elif 'weight' in column_name.lower():
                return round(random.uniform(0.01, 1.0), 2)
            elif 'metric_value' in column_name.lower():
                return round(random.uniform(-1000, 1000), 2)
            elif 'id' in column_name.lower():
                return random.randint(1000, 9999)
            else:
                return random.randint(100, 10000)
        elif db_type == 'text':
            # Generate sample text values
            text_options = ['AAA', 'BBB', 'CCC', 'XYZ', 'TEST', 'PROD', 'DEV']
            if 'code' in column_name.lower():
                return random.choice(text_options)
            elif 'greek' in column_name.lower():
                return random.choice(['delta', 'gamma', 'theta', 'vega', 'rho'])
            else:
                return random.choice(text_options)
        elif db_type == 'date':
            year = random.randint(2020, 2024)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            return f"{year}-{month:02d}-{day:02d}"
        elif db_type == 'timestamp':
            year = random.randint(2020, 2024)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00"
        return None
    
    def _generate_simple_query(self) -> Tuple[str, Dict]:
        """Generate a simple query with 1 filter."""
        # Pick a random table
        table = random.choice(self.tables)
        
        # Pick a random column from that table
        columns = self.columns_by_table[table]
        column = random.choice(columns)
        
        # Get an alias/real_name for this column
        key = (table, column)
        if key not in self.aliases_by_column or not self.aliases_by_column[key]:
            return self._generate_simple_query()  # Retry
        
        term_info = random.choice(self.aliases_by_column[key])
        db_type = self.column_types[key]
        
        # Generate query text
        template = random.choice(self.simple_templates)
        use_alias = random.choice([True, False])
        term = term_info['alias'] if use_alias else term_info['real_name']
        
        query = template.format(alias=term, real_name=term)
        
        # Generate filter
        operator = random.choice(self.operators_by_type.get(db_type, ['=']))
        value = self._get_random_value_for_type(db_type, column)
        
        filter_dict = {
            "table": table,
            "filters": [
                {
                    "column": column,
                    "op": operator,
                    "value": value
                }
            ]
        }
        
        return query, filter_dict
    
    def _generate_medium_query(self) -> Tuple[str, Dict]:
        """Generate a medium complexity query with 2-3 filters."""
        # Pick a random table
        table = random.choice(self.tables)
        
        # Pick 2-3 random columns from that table
        columns = self.columns_by_table[table]
        num_filters = random.choice([2, 3])
        if len(columns) < num_filters:
            return self._generate_medium_query()  # Retry with different table
        
        selected_columns = random.sample(columns, num_filters)
        
        # Get an alias/real_name for the first column
        key = (table, selected_columns[0])
        if key not in self.aliases_by_column or not self.aliases_by_column[key]:
            return self._generate_medium_query()  # Retry
        
        term_info = random.choice(self.aliases_by_column[key])
        use_alias = random.choice([True, False])
        term = term_info['alias'] if use_alias else term_info['real_name']
        
        # Generate query text (simplified)
        query_parts = [f"show me {term} {table}"]
        filters = []
        
        for i, column in enumerate(selected_columns):
            key = (table, column)
            db_type = self.column_types[key]
            operator = random.choice(self.operators_by_type.get(db_type, ['=']))
            value = self._get_random_value_for_type(db_type, column)
            
            filters.append({
                "column": column,
                "op": operator,
                "value": value
            })
            
            # Add to query text
            if i > 0:
                if operator in ['>', '>=']:
                    query_parts.append(f"{column} over {value}")
                elif operator in ['<', '<=']:
                    query_parts.append(f"{column} under {value}")
                elif operator == '=':
                    query_parts.append(f"{column} equals {value}")
        
        query = " with ".join(query_parts)
        
        filter_dict = {
            "table": table,
            "filters": filters
        }
        
        return query, filter_dict
    
    def _generate_complex_query(self) -> Tuple[str, Dict]:
        """Generate a complex query with 4+ filters."""
        # Pick a random table
        table = random.choice(self.tables)
        
        # Pick 4-6 random columns from that table
        columns = self.columns_by_table[table]
        num_filters = random.choice([4, 5, 6])
        if len(columns) < num_filters:
            num_filters = len(columns)
        
        selected_columns = random.sample(columns, min(num_filters, len(columns)))
        
        # Get an alias/real_name for the first column
        key = (table, selected_columns[0])
        if key not in self.aliases_by_column or not self.aliases_by_column[key]:
            return self._generate_complex_query()  # Retry
        
        term_info = random.choice(self.aliases_by_column[key])
        use_alias = random.choice([True, False])
        term = term_info['alias'] if use_alias else term_info['real_name']
        
        # Generate query text (simplified)
        query_parts = [f"find {term} {table} where"]
        filters = []
        conditions = []
        
        for column in selected_columns:
            key = (table, column)
            db_type = self.column_types[key]
            operator = random.choice(self.operators_by_type.get(db_type, ['=']))
            value = self._get_random_value_for_type(db_type, column)
            
            filters.append({
                "column": column,
                "op": operator,
                "value": value
            })
            
            # Add to query text
            if operator in ['>', '>=']:
                conditions.append(f"{column} over {value}")
            elif operator in ['<', '<=']:
                conditions.append(f"{column} under {value}")
            elif operator == '=':
                conditions.append(f"{column} is {value}")
            elif operator == 'LIKE':
                conditions.append(f"{column} like {value}")
        
        query = query_parts[0] + " " + ", ".join(conditions)
        
        filter_dict = {
            "table": table,
            "filters": filters
        }
        
        return query, filter_dict
    
    def generate_dataset(self, size: int, distribution: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> List[Dict]:
        """
        Generate a dataset of query-filter pairs.
        
        Args:
            size: Total number of examples to generate
            distribution: Tuple of (simple_ratio, medium_ratio, complex_ratio)
        
        Returns:
            List of dictionaries with 'query' and 'filter' keys
        """
        simple_ratio, medium_ratio, complex_ratio = distribution
        
        num_simple = int(size * simple_ratio)
        num_medium = int(size * medium_ratio)
        num_complex = size - num_simple - num_medium
        
        dataset = []
        
        # Generate simple queries
        for _ in tqdm(range(num_simple), desc="Generating simple queries"):
            query, filter_dict = self._generate_simple_query()
            dataset.append({"query": query, "filter": filter_dict})
        
        # Generate medium queries
        for _ in tqdm(range(num_medium), desc="Generating medium queries"):
            query, filter_dict = self._generate_medium_query()
            dataset.append({"query": query, "filter": filter_dict})
        
        # Generate complex queries
        for _ in tqdm(range(num_complex), desc="Generating complex queries"):
            query, filter_dict = self._generate_complex_query()
            dataset.append({"query": query, "filter": filter_dict})
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        return dataset
    
    def validate_filter(self, filter_dict: Dict) -> bool:
        """
        Validate that a filter dictionary is well-formed.
        
        Args:
            filter_dict: The filter dictionary to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required keys
            if 'table' not in filter_dict or 'filters' not in filter_dict:
                return False
            
            # Check table is valid
            if filter_dict['table'] not in self.tables:
                return False
            
            # Check filters is a list
            if not isinstance(filter_dict['filters'], list):
                return False
            
            # Check each filter
            for f in filter_dict['filters']:
                if 'column' not in f or 'op' not in f or 'value' not in f:
                    return False
                
                # Validate column exists for the table
                if f['column'] not in self.columns_by_table[filter_dict['table']]:
                    return False
            
            return True
        except Exception:
            return False


def split_dataset(dataset: List[Dict], train_ratio: float, val_ratio: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (remainder goes to test)
    
    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:]
    
    return train_set, val_set, test_set


def write_jsonl(data: List[Dict], filepath: Path):
    """
    Write data to JSONL format.
    
    Args:
        data: List of dictionaries to write
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic derivatives query-filter dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory path for generated files'
    )
    parser.add_argument(
        '--size',
        type=int,
        required=True,
        help='Total number of examples to generate'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Proportion for training set (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Proportion for validation set (default: 0.15)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default='Derivatives_Lingo.csv',
        help='Path to Derivatives_Lingo.csv (default: Derivatives_Lingo.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train_ratio + val_ratio must be less than 1.0")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading schema from {args.csv_path}...")
    generator = DerivativesDataGenerator(args.csv_path)
    
    print(f"\nGenerating {args.size} query-filter pairs...")
    dataset = generator.generate_dataset(args.size)
    
    print("\nValidating generated filters...")
    valid_count = 0
    for item in tqdm(dataset, desc="Validating"):
        if generator.validate_filter(item['filter']):
            valid_count += 1
    
    print(f"Validation: {valid_count}/{len(dataset)} filters are well-formed")
    
    print("\nSplitting into train/val/test sets...")
    train_set, val_set, test_set = split_dataset(dataset, args.train_ratio, args.val_ratio)
    
    print(f"Train: {len(train_set)} examples")
    print(f"Validation: {len(val_set)} examples")
    print(f"Test: {len(test_set)} examples")
    
    print("\nWriting JSONL files...")
    write_jsonl(train_set, output_dir / 'train.jsonl')
    write_jsonl(val_set, output_dir / 'val.jsonl')
    write_jsonl(test_set, output_dir / 'test.jsonl')
    
    print(f"\n✓ Successfully generated dataset in {output_dir}")
    print(f"  - train.jsonl: {len(train_set)} examples")
    print(f"  - val.jsonl: {len(val_set)} examples")
    print(f"  - test.jsonl: {len(test_set)} examples")


if __name__ == '__main__':
    main()
