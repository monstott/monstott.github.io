---
layout: post
title:      "A Walk Through the Monty Hall"
date:       2020-01-31 21:56:02 +0000
permalink:  a_walk_through_the_monty_hall
---

### Motivation.
The Monty Hall problem is a probability puzzle based on the television game show [Let's Make a Deal](https://www.cbs.com/shows/lets_make_a_deal/). It is named after its original host [Monty Hall](https://en.wikipedia.org/wiki/Monty_Hall). This problem became famous in 1990 after causing a great deal of confusion following its appearance in a column in [*Parade Magazine*](http://people.math.harvard.edu/~knill/oldtexas/Teaching/MA464/PUZZLES/marilyn.gameshow.html). The author of the column and the solution to this problem, Marilyn vos Savant, is quoted as saying:

> I'm receiving thousands of letters, nearly all insisting that I'm wrong. Of the letters from the general public, 92% are against my answer; and of the letters from universities, 65% are against my answer. Overall, nine out of 10 readers completely disagree with my reply.

This post is written to achieve an understanding of the solution to this problem that has vexed so many. 

### Problem Situation.

Here's the setup of the Monty Hall problem: 
* You are a contestant on a game show. 
* The host offers you three doors to choose from. 
* Behind two of the doors are unwanted items (e.g., goats) and behind the reamining door is the real prize (e.g., a car).
* You choose a door.
* Instead of opening your chosen door, the host opens one of the two doors you did not select.
* The host then asks if you want to stay with your door or switch your choice to the other closed door. 

Here's the puzzling question:
* Do you switch doors?

Additionally, here are the assumptions made in this problem:

* The host must offer the chance to switch doors.
* The host must open a door that was not picked by you.
* The host must open a door that does not have the prize.

### Incorrect Solution.

 The wrong interpretation of the solution to this problem looks something like this: probability says there is a **⅓** chance of choosing the prize door out of the 3 doors in the beginning. Now that one of the doors has been opened, there is a **½** chance of choosing the prize door from the remaining 2 doors. Using this logic, switching doors does not increase the chance of winning the prize. This line of thinking, however, is incorrect.

###  Correct Solution from Scenarios.

Suppose the 3 doors are colored Red, Green, and Blue.  Now suppose that the prize is in **Green Door**.

##### Here are the scenarios: 

If you choose the **Green Door**, the host must open either the **Blue Door** or the **Red Door**. 
* If, after opening one of the other doors, you decide to switch then you *Lose*. (1)
* If, after opening one of the other doors, you decide to stay then you *Win*. (2)

If you choose the **Blue Door**, the host must open the **Red Door** since the prize is behind the **Green Door**.  
* If, after opening the **Red Door**, you decide to switch then you *Win*. (3)
* If, after opening the **Red Door**, you decide to stay then you *Lose*. (4)

If you choose the **Red Door** the must open the **Blue Door** since the prize is behind the **Green Door**.
* If, after opening the **Blue Door**, you decide to switch then you *Win*. (5)
* If, after opening the **Blue Door**, you decide to stay then you *Lose*. (6)

##### Looking  at the results by decision:
* If you decide to switch, the probability of winning the prize is **⅔** (scenarios 1, 3, and 5). 
* If you decide to stay, the probability of winning the prize is **⅓** (scenarios 2, 4, and 6).

Switching doors doubles your chances of winning. This answer can also be viewed through the lens of probability.

### Correct Solution from Probability.

This is the initial probability that the prize is behind each door: 

* P(Prize in **Red**) = P(Prize in **Green**) = P(Prize in **Blue**) = **⅓**

If you choose the **Green Door**, the probability that the host will open the **Red Door** becomes:

* P(Open **Red** | Prize in **Red**) = **0**
* P(Open **Red** | Prize in **Green**)  = **½**
* P(Open **Red** | Prize in **Blue**) = **1**

Suppose that the host opens the **Red Door**. What is the probability that the remaining doors have the prize?

##### Probability if you switch:
[Bayes' Theorem](http://www.hep.upenn.edu/~johnda/Papers/Bayes.pdf) can be used to determine the probability of this event occuring using relevant information on event conditions. 

The probability the prize is behind the **Blue Door** given that the **Red Door** is opened and you start with the **Green Door**  (i.e., you switch):

* P(Prize in **Blue** | Open **Red**) = P(Open **Red** | Prize in **Blue**) **⋅** P(Prize in **Blue**) **÷** P(Open **Red**)

The [law of total probability](https://www.probabilitycourse.com/chapter1/1_4_2_total_probability.php) is used to relate the marginal probability that the **Red Door** is opened to conditional probabilities:

* P(Prize in **Blue** | Open **Red**) = *Numerator* **÷** *Denominator*, where
 * *Numerator* = P(Open **Red** | Prize in **Blue**) **⋅** P(Prize in **Blue**) = **1 ⋅ ⅓** = **⅓**
 * *Denominator* = P(Open **Red** | Prize in **Red**) **⋅** P(Prize in **Red**) + P(Open **Red** | Prize in **Green**) **⋅** P(Prize in **Green**) + P(Open **Red** | Prize in **Blue**) **⋅** P(Prize in **Blue**) = **0 ⋅ ⅓ + ½ ⋅ ⅓ + 1 ⋅ ⅓** = **½**
* P(Prize in **Blue** | Open **Red**) = **⅓** **÷** **½** = **⅔**

##### Probability if you stay:
The probability the prize is behind the **Green Door** given that the **Red Door** is opened and you start with the **Green Door** (i.e., you stay) :

* P(Prize in **Green** | Open **Red**) = P(Open **Red** | Prize in **Green**) **⋅** P(Prize in **Green**) **÷** P(Open **Red**)
* P(Prize in **Green** | Open **Red**) = *Numerator* **÷** *Denominator*, where
 * *Numerator* = P(Open **Red** | Prize in **Green**) **⋅** P(Prize in **Green**) = **½ ⋅ ⅓** = **⅙**
 * *Denominator* = P(Open **Red** | Prize in **Red**) **⋅** P(Prize in **Red**) + P(Open **Red** | Prize in **Green**) **⋅** P(Prize in **Green**) + P(Open **Red** | Prize in **Blue**) **⋅** P(Prize in **Blue**) = **0 ⋅ ⅓ + ½ ⋅ ⅓ + 1 ⋅ ⅓** = **½**
* P(Prize in **Blue** | Open **Red**) = **⅙** **÷** **½** = **⅓**

This solution shows that switching doors doubles the chances of winning the prize over staying with the initial door choice. The fact that a door without the prize is opened improves the chances of winning because this additional information can inform the contestant's decision. 

### Solving the Problem with Python.

The first step in implementing the Monty Hall Problem in python is to create some doors. A  `Make_Door` class is constructed with the ability to create closed doors that do not contain the prize.

```
class DoorCreation(object):
    '''
    PURPOSE: This class creates doors.
    ATTRIBUTES: 
        has_prize = if the door has the prize
        is_open = if the door has been opened
    '''
    has_prize = False
    is_open = False
```

The second step is to assign the prize to one of the created doors. This is done randomly in the `selectDoor()` method of the `MontyHall` class. The third step is to open the door that is not selected by the contestant and does not have the prize. This is performed in the `openDoor()` method. The fourth step is to offer the contestant a chance to switch their selected door to the remaining closed door. This offer is randomly taken in the `switchDoor()` method. The final step is to setup games to perform the all of the actions in the Monty Hall Problem and return the results of each game. The `@classmethod` decorator method `makeDeal()` takes care of this.

```
import random

class MontyHall(object):
    '''
    PURPOSE: This class instantiates 3 doors and selects one at random to have the prize.
    ATTRIBUTES:
        selected_door = the door selected by the contestant
        if_switch = if the contestant will switch doors
    METHODS:
        selectDoor = this method randomly selects one of the doors for the contestant
        openDoor = this method opens the door that is not selected by the player and does not have the prize
        switchDoor = this method randomly switches the contestant door to the remaining door
        makeDeal = this decorator method performs the actions of the uninstantiated class methods and returns the results
    '''
    selected_door = None
    if_switch = False

    def __init__(self, door_count=3):
        self.doors = []
        for i in range(door_count): 
            self.doors.append(DoorCreation())
        random.choice(self.doors).has_prize = True

    def selectDoor(self):
        self.selected_door = random.choice(self.doors)

    def openDoor(self):
        other_doors = [door for door in self.doors if door is not self.selected_door]
        for door in other_doors: 
            if not door.has_prize:
                door.is_open = True
                break

    def switchDoor(self):
        last_door = [door for door in self.doors if door is not self.selected_door
                                                          and not door.is_open][0]
        if random.randint(0, 1):
            self.if_switch = True
            self.selected_door = last_door
            
    @classmethod
    def makeDeal(cls):
        player = cls()
        player.selectDoor()
        player.openDoor()
        player.switchDoor()
        
        return data(player.selected_door.has_prize, player.if_switch)  
```

Now that the problem is ready to run, I'll create 100,000 games of the Monty Hall Problem and look at how switching doors affects the outcome.

```
# Monty Hall Problem results
from collections import namedtuple

data = namedtuple('data', ['if_win', 'if_switch'])
num_games = 100000
results = [MontyHall.makeDeal() for i in range(num_games)]

switch_win = list(filter(lambda x: x.if_win and x.if_switch, results))
switch_lose = list(filter(lambda x: not x.if_win and x.if_switch, results))
stay_win = list(filter(lambda x: x.if_win and not x.if_switch, results))
stay_lose = list(filter(lambda x: not x.if_win and not x.if_switch, results))

print("Monty Hall Problem Results")
print("-"*30)
print("Number of Game:", num_games)
print("")
print("Switch + Win:", len(switch_win))
print("Switch + Lose:", len(switch_lose))
print("Switch Success Ratio: {0:.2f}%".format(len(switch_win) / (len(switch_win) + len(switch_lose)) * 100))
print("")
print("Stay + Win:", len(stay_win))
print("Stay + Lose:", len(stay_lose))
print("Stay Success Ratio: {0:.2f}%".format(len(stay_win) / (len(stay_win) + len(stay_lose)) * 100))

> Monty Hall Problem Results
> ------------------------------
> Number of Game: 100000
> 
> Switch + Win: 33493
> Switch + Lose: 16596
> Switch Success Ratio: 66.87%
> 
> Stay + Win: 16517
> Stay + Lose: 33394
> Stay Success Ratio: 33.09%
```

If the contestant switches doors their chance of winning is 66%. This compares with the 33% chance of winning if the contestant chooses to stay with their original door.  The 2-to-1 ratio of winning-to-losing if the door is switched that was discussed in the scenario and probability solutions is confirmed by the Python experiment. 

### Final thoughts.
This investigation has shown me how an initial easy answer can be proven wrong after additional thought. The introduction of additional information that occurs within this problem is subtle. Realizing this fact makes for a fun challenge to an interesting problem. With luck, this walkthrough will help prevent others and I from being duped by similar probability puzzles in the future.


