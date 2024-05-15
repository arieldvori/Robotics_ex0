## Requirements
- Python 3.x
- Required libraries: `csv`, `datetime`, `pandas`, `numpy`, `navpy`, `gnssutils`, `simplekml`, `warnings`

## Usage
1. **Download the Code**: Clone or download the repository to your local machine.
2. **Navigate to Directory**: Open a terminal or command prompt and navigate to the directory where the code is located.
3. **Run the Program**:
    - The main program file is `parsing_and_xml.py`.
    - You need to specify the input file containing GNSS measurements.
    - Run the program by executing the following command:
    ```bash
    python parsing_and_xml.py
    ```
4. **Input the Wanted File**:
    - After running the program, you'll be prompted to input the type of data file you want to process: `driving`, `walking`, or `fixed`.
    - Choose the desired option by typing one of these keywords and pressing Enter.
5. **Output Files**:
    - The program will generate two output files:
        - A CSV file containing processed GNSS data named `<filename>-output.csv`.
        - A KML file containing GPS positions named `<filename>-output.kml`.

## Example
```bash
$ python parsing_and_xml.py
Enter driving/walking/fixed: walking
```
