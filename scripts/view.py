import sqlite3

def view_feedback():
    # Connect to the database
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()

    # Execute query
    cursor.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()

    # Print results
    print("Last 5 Feedback Entries:")
    print("-" * 50)
    for row in rows:
        print(row)

    # Close connection
    conn.close()

if __name__ == "__main__":
    view_feedback()
