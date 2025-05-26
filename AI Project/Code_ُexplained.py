import matplotlib.pyplot as plt

# Example data
egypt_data = {'Deaths': [10, 20, 15, 25, 30, 100, 5, 15, 25, 10]}

# Box plot
plt.boxplot(egypt_data['Deaths'], 
            vert=False, 
            patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='blue'), 
            whiskerprops=dict(color='blue'), 
            flierprops=dict(markerfacecolor='red', marker='o'))
plt.title('Box Plot for Deaths in Egypt')
plt.xlabel('Number of Deaths')
plt.show()
