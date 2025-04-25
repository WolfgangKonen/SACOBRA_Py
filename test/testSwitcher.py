def switch_demo(argument):
    switcher = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    print (switcher.get(argument, "Invalid month"))


def one():
    return "January"


def two():
    return "February"


def three():
    return "March"

def numbers_to_months(argument):
    switcher = {
        1: one,
        2: two,
        3: three,
    }
    # Get the function from switcher dictionary
    func = switcher.get(argument, lambda: "Invalid month")
    # Execute the function
    print(func())

#switch_demo(13)

numbers_to_months(2)