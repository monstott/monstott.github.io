---
layout: post
title:      "Visualizing Set Diagrams with Python"
date:       2020-01-24 22:30:27 +0000
permalink:  visualizing_set_diagrams_with_python
---

## Motivation for this post.

One of the major challenges faced by data scientists is how to effectively communicate results that are obtained. Often, the audience on the receiving end of a presentation is composed of executive stakeholders without a background in the technical fields required for a particular project. As a result, clear and uncomplicated explanations are imperative in order that correct interpretations are made from information delivered by the presentation. Of all of the formats and methods available to choose from, visualizations are a primary ally for communicating effectively with others. Surprisingly, one of the simplest and most tried-and-true visualizations is not native to the **Matplotlib** library for *Python*. That simple visualization is the Venn diagram. The Venn diagram is a building block of set theory and a favorite tool for displaying the relationships between different entitites. In order to make this visualization available to a data scientist's toolkit, this blog entry will examine how to build Venn diagrams with *Python*.

## Identifying a solution.

The functions necessary to build this visualization are found within a third-party package that extends the functionality of **Matplotlib**. This solution is built using Jupyter Notebook. To begin, install the package:

```
!easy_install matplotlib-venn

# or

!pip install matplotlib-venn
```

It should be noted that the **Matplotlib-Venn** third-party package is dependent on the **Numpy**, **Scipy**, and **Matplotlib** libraries. If they are not available, they may be installed in a similar manner. 

## Taking a look at the functions.

The package has six main functions: **venn2**, **venn2_unweighted**, **venn2_circles**, **venn3**, **venn3_unweighted**, and **venn3_circles**. I'll examine each of them and discuss their capabilities.

###  The *venn2* function.

This function  has one required input argument. The argument can either be a *3*-element list containing subset sizes; a *3*-element dictionary with subset size values for the keys  '10', '01', and '11'; or, a *2*-element list containing set objects. 

For the first method,
1. the *first* element is the size of the set on the left
2. the *second* element is the size of the set on the right
3. and the *third* element is the size of the overlap between the two sets.

For the second method,
1. the '10' key element is the size of the set on the left
2. the '01' key element is the size of the set on the right
3. and the '11' key element is the size of the overlap between the two sets.

For the third method, 
1. the *first* element is the list of objects for the set on the left 
2. and the *second* element is the list of objects for the set on the right. 
Objects present in both sets will be placed in the overlap between the two sets.

```
from matplotlib_venn import venn2


venn2(subsets = (2, 4, 1)) # 3-element list of subset sizes

# or 

venn2(subsets = {'10': 2, '01': 4, '11': 1}) # 3-element dictionary of subset sizes

# or 

venn2(subsets = [set(['A', 'B', 'C']), set(['C', 'D', 'E', 'F', 'G'])]) # 2-element list of set objects
```

![venn2 Example](https://github.com/monstott/Blogs/raw/master/venn2example.png)


The output from this function is a colored and annotated two-circle Venn diagram in which the area of each shape is proportional to the number of elements it contains. In the example above, the region for subset **A** is half the size of **B** and twice the size of the overlap **A and B**. The region for subset **B** is twice the size of **A** and four times the size of the overlap **A and B**. 

### The *venn2_unweighted* function.

This function has the same required input argument as the **venn2** function. Its output is different. The output of this function does not make the area of each shape proportional to the number of elements. Instead, the area of each shape is the same.

```
from matplotlib_venn import venn2_unweighted


venn2_unweighted(subsets = (2, 4, 1))  # 3-element list of subset sizes

# or

venn2_unweighted(subsets = {'10': 2, '01': 4, '11': 1}) # 3-element dictionary of subset sizes

# or

venn2_unweighted(subsets = [set(['A', 'B', 'C']), set(['C', 'D', 'E', 'F', 'G'])]) # 2-element list of subset sizes
```
![venn2_unweighted Example](https://github.com/monstott/Blogs/raw/master/venn2unweightedexample.png)

### The *venn2_circles* function.

This function has the same required input argument as the **venn2** function. Its output is different. The output of this function removes the coloring and text annotation from the **venn2** function, returning only the set circles and their relationship.

```
from matplotlib_venn import venn2_circles


venn2_circles(subsets = (2, 4, 1)) # 3-element list of subset sizes

# or 

venn2_circles(subsets = {'10': 2, '01': 4, '11': 1}) # 3-element dictionary of subset sizes

# or

venn2_circles(subsets = [set(['A', 'B', 'C']), set(['C', 'D', 'E', 'F', 'G'])]) # 2-element list of set objects
```

![venn2_circles Example](https://github.com/monstott/Blogs/raw/master/venn2circlesexample.png)

### The *venn3* function.

This function  has one required input argument. The argument can either be a *7*-element list containing subset sizes; a *7*-element dictionary containing subset size values for the keys '100', '010', '001', '110', '101', '011', and '111'; or, a *3*-element list containing set objects. 

For the first method, 
1. the *first* element is the size of the set on the left
2. the *second* element is the size of the set on the right
3. the *third* element is the size of the overlap between the left and right sets
4. the *fourth* element is the size of the set on the bottom
5. the *fifth* element is the size of the overlap between the left and bottom sets
6. the *sixth* element is the size of the overlap between the right and bottom sets
7. and the *seventh* element is the size of the overlap between all three sets.

For the second method,
1. the '100' key element is the size of the set on the left
2. the '010' key element is the size of the set on the right
3. the '001' key element is the size of the set on the bottom
4. the '110' key element is the size of the overlap between the left and right sets
5. the '101' key element is the size of the overlap between the left and the bottom sets
6. the '011' key element is the size of the overlap between the right and the bottom sets
7. and the '111' key element is the size of the overlap between all three sets.

For the third method, 
1. the *first* element is the list of objects for the set on the left
2. the *second* element is the list of objects for the set on the right
3. and the *third* element is the list of objects for the set on the bottom.
Objects present in more than one set will be placed in the appropriate overlap section.

```
from matplotlib_venn import venn3


venn3(subsets = (4, 6, 1, 8, 2, 3, 0)) # 7-element list of subset sizes

# or

venn3(subsets = {'100': 4, '010': 6, '110': 1, '001': 8, '101': 2, '011': 3, '111': 0}) # 7-element dictionary of subset sizes

# or 

venn3([set(['A', 'B', 'C', 'D', 'E', 'F', 'G']), 
       set(['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']), 
       set(['A', 'B', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'])]) # 3-element list of set objects
```
![venn3 Example](https://github.com/monstott/Blogs/raw/master/venn3example.png)

The output from this function is a colored and annotated three-circle Venn diagram in which the area of each shape is proportional to the number of elements it contains. In the example above, the region for subset **A** is two-thirds the size of **B** and half the size of **C**. The size of the overlap **A and B** is 1. The size of the overlap **A and C** is  twice that size and the size of the overlap **B and C** is three times the size. There is no overlap **A and B and C** between all the sets. 

### The *venn3_unweighted* function.
​
This function has the same required input argument as the **venn3** function. Its output is different. The output of this function does not make the area of each shape proportional to the number of elements. Instead, the area of each shape is the same.
​
```
from matplotlib_venn import venn3_unweighted

​
venn3_unweighted(subsets = (4, 6, 1, 8, 2, 3, 0)) # 7-element list of subset sizes

# or

venn3_unweighted(subsets = {'100': 4, '010': 6, '110': 1, '001': 8, '101': 2, '011': 3, '111': 0}) # 7-element dictionary of subset sizes
​
# or
​
venn3_unweighted([set(['A', 'B', 'C', 'D', 'E', 'F', 'G']), 
                  set(['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']), 
                  set(['A', 'B', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'])]) # 3-element list of subset sizes
```
![venn3_unweighted Example](https://github.com/monstott/Blogs/raw/master/venn3unweightedexample.png)
​

### The *venn3_circles* function.

This function has the same required input argument as the **venn3** function. Its output is different. The output of this function removes the coloring and text annotation from the **venn3** function, returning only the set circles and their relationships.

```
from matplotlib_venn import venn3_circles

venn3_circles(subsets = (4, 6, 1, 8, 2, 3, 0)) # 7-element list of subset sizes

# or 

venn3_circles(subsets = {'100': 4, '010': 6, '110': 1, '001': 8, '101': 2, '011': 3, '111': 0}) # 7-element dictionary of subset sizes

# or

venn3_circles([set(['A', 'B', 'C', 'D', 'E', 'F', 'G']), 
               set(['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']), 
               set(['A', 'B', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'])]) # 3-element list of set objects
```

![venn3_circles Example](https://github.com/monstott/Blogs/raw/master/venn3circlesexample.png)

The **venn2** and **venn3** functions return an object with class VennDiagram. This object has methods that provide access to additonal information and functionality.

## Investigating the *VennDiagram* class.
The example used to illustrate the capabilities of this class is a modified version of the **venn3** example. Note how an optional input parameter *set_labels* has been used to change the label text.

```
v = venn3(subsets = (4, 6, 1, 8, 2, 3, 0), set_labels = ('First', 'Second', 'Third'))
```

![VennDiagram class Example](https://github.com/monstott/Blogs/raw/master/VennDiagramclass.png)

The object returned from **venn2** and **venn3** has the following methods and attributes: 

* **centers**: returns (x, y) coordinates for the centers of all circles in the diagram.

```
v.centers

> array([[-0.28856901,  0.17774299],
       [ 0.23356695,  0.17774299],
       [-0.02428357, -0.23243314]])
```

* **get_circle_center**:  returns the (x, y) coordinates for the center of one circle; the input parameter is 0, 1 or 2.

```
v.get_circle_center(0)
		
> [-0.28856901  0.17774299]
```

* **get_circle_radius**:  returns the radius for one circle; the input parameter is 0, 1 or 2.

```
v.get_circle_radius(0)
		
> 0.30469719964297715
```

* **get_label_by_id**:  returns a subset label by one region ID; region ID is '10', '01' or '11' for a 2-circle diagram and '100', '110', etc., for a 3-circle diagram.

```
v.get_label_by_id('100')

> Text(-0.4399821131834529, 0.2931018585930942, '4')
```

* **get_patch_by_id** : returns a patch by a region ID.

```
v.get_patch_by_id('100')

> PathPatch37((-0.0656029, 0.385412) ...)
```

* **hide_zeroes**:  hides the labels for subsets with a size of zero.

```
v.hide_zeroes
```

![hide_zeroes Method Example](https://github.com/monstott/Blogs/raw/master/hide_zeroes.png)

* **id2idx**: returns a dictionary of region IDs and subset sizes.

```
v.id2idx

> {'10': 0,
 '01': 1,
 '11': 2,
 '100': 0,
 '010': 1,
 '110': 2,
 '001': 3,
 '101': 4,
 '011': 5,
 '111': 6,
 'A': 0,
 'B': 1,
 'C': 2}
```

* **patches**: returns a list of objects of class PathPatch.

```
v.patches

> [<matplotlib.patches.PathPatch at 0x26e858f6048>,
 <matplotlib.patches.PathPatch at 0x26e858f6160>,
 <matplotlib.patches.PathPatch at 0x26e858f6278>,
 <matplotlib.patches.PathPatch at 0x26e858f6390>,
 <matplotlib.patches.PathPatch at 0x26e858f64a8>,
 <matplotlib.patches.PathPatch at 0x26e858f65c0>,
 <matplotlib.patches.PathPatch at 0x26e858f66d8>]
```

* **radii**: returns the radius for all circles.

```
v.radii

> array([0.3046972 , 0.36418281, 0.41523229])
```

* **set_labels**: returns the (x, y) coordinates and text label for each set.

```
v.set_labels

> [Text(-0.44091761338207724, 0.4824401891925971, 'First'),
 Text(0.41565835558170994, 0.5419257997469797, 'Second'),
 Text(-0.024283569992783027, -0.6891886596539031, 'Third')]
```

* **subset_labels**: returns the (x, y) coordinates and label for each subset size.

```
v.subset_labels

> [Text(-0.4399821131834529, 0.2931018585930942, '4'),
 Text(0.388108112645034, 0.31949095859751864, '6'),
 Text(-0.056420507321689614, 0.2651473041660194, '1'),
 Text(-0.04669136159557198, -0.4376932281000442, '8'),
 Text(-0.23290178929168304, -0.01583292369428826, '2'),
 Text(0.14977594058744756, -0.027289566147648513, '3'),
 Text(-0.052744164225607106, 0.09049160890186, '0')]
```

Using functionality from **venn3**  objects can produce enhanced and appealing visualizations.

## Putting it all together.
Using the Venn diagram package and adjusting the settings of arguments in objects returned by its functions can improve the quality of a diagram intended for presentation.
```
from matplotlib import pyplot as plt

plt.figure(figsize=(5, 5))

v = venn3(subsets = (4, 6, 1, 8, 2, 3, 0), set_labels = ('First', 'Second', 'Third'))
v.get_patch_by_id('100').set_alpha(0.1)
v.get_patch_by_id('100').set_facecolor('Orchid')
v.get_patch_by_id('100').set_edgecolor('MediumVioletRed')
v.get_patch_by_id('100').set_linestyle('--') # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
v.get_patch_by_id('100').set_linewidth(5)
v.get_patch_by_id('100').set_hatch('*') # {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
v.get_label_by_id('100').set_text('One Item')
v.get_label_by_id('A').set_text('Alpha')

v.get_patch_by_id('010').set_alpha(0.6)
v.get_patch_by_id('010').set_facecolor('Salmon')
v.get_patch_by_id('010').set_edgecolor('RoyalBlue')
v.get_patch_by_id('010').set_linestyle('-.')
v.get_patch_by_id('010').set_linewidth(5)
v.get_patch_by_id('010').set_hatch('/') 
v.get_label_by_id('010').set_text('Six Items')
v.get_label_by_id('B').set_text('Beta')

v.get_patch_by_id('001').set_alpha(0.8)
v.get_patch_by_id('001').set_facecolor('LightSeaGreen')
v.get_patch_by_id('001').set_edgecolor('LightSlateGray')
v.get_patch_by_id('001').set_linestyle(':') 
v.get_patch_by_id('001').set_linewidth(5)
v.get_patch_by_id('001').set_hatch('.')
v.get_label_by_id('001').set_text('Eight Items')
v.get_label_by_id('C').set_text('Gamma')

v.get_patch_by_id('110').set_alpha(0.4)
v.get_patch_by_id('110').set_facecolor('GoldenRod')
v.get_patch_by_id('110').set_edgecolor('Lavender')
v.get_patch_by_id('110').set_linestyle('-') 
v.get_patch_by_id('110').set_linewidth(5)

v.get_patch_by_id('011').set_alpha(0.4)
v.get_patch_by_id('011').set_facecolor('CornflowerBlue')
v.get_patch_by_id('011').set_edgecolor('DarkSlateBlue')
v.get_patch_by_id('011').set_linestyle('-')
v.get_patch_by_id('011').set_linewidth(5)

v.get_patch_by_id('101').set_alpha(0.4)
v.get_patch_by_id('101').set_facecolor('RosyBrown')
v.get_patch_by_id('101').set_edgecolor('Plum')
v.get_patch_by_id('101').set_linestyle('-') 
v.get_patch_by_id('101').set_linewidth(5)

v.get_patch_by_id('111').set_alpha(0.4)
v.get_patch_by_id('111').set_facecolor('SeaGreen')
v.get_patch_by_id('111').set_edgecolor('SteelBlue')
v.get_patch_by_id('111').set_linestyle('-') 
v.get_patch_by_id('111').set_linewidth(5)
v.get_label_by_id('111').set_text('[]')

plt.title('Venn Diagram in Python')
plt.show()
```

![Improved Venn Diagram Example](https://github.com/monstott/Blogs/raw/master/VennDiagramfinal.png)


## More information on object classes.
The **venn2_circles** and **venn3_circles** functions return a list of objects of class matplotlib.patch.Circle.  The **patches**, **get_label_by_id**, and **get_patch_by_id** methods from **venn2** and **venn3** return objects of class matplotlib.patches.PathPatch. Objects from these classes have a wide selection of methods and attributes that provide an enhaced ability to tailor diagrams. 

Details may be reviewed in detail within the Matplotlib *Circle* [documentation](https://matplotlib.org/api/_as_gen/matplotlib.patches.Circle.html) and *PathPatch* [documentation](https://matplotlib.org/api/_as_gen/matplotlib.patches.PathPatch.html#matplotlib.patches.PathPatch).

## Closing thoughts.
Set diagrams are one way to visually represent the relationships between different entities. This is possible in *Python* with the help of third-party packages. This blog post has taken an in-depth look at the functionality of one available package. A breakdown of the inputs, outputs, and class dependencies of its functions has made it possible to use this tool in future projects and stakeholder presentations.  

