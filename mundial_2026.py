import pandas as pd

data = [
    ["Francia", 1, "UEFA", "I"], ["España", 2, "UEFA", "H"], ["Argentina", 3, "CONMEBOL", "J"],
    ["Inglaterra", 4, "UEFA", "L"], ["Portugal", 5, "UEFA", "K"], ["Brasil", 6, "CONMEBOL", "C"],
    ["Países Bajos", 7, "UEFA", "F"], ["Marruecos", 8, "CAF", "C"], ["Bélgica", 9, "UEFA", "G"],
    ["Alemania", 10, "UEFA", "E"], ["Croacia", 11, "UEFA", "L"], ["Colombia", 13, "CONMEBOL", "K"],
    ["Senegal", 14, "CAF", "I"], ["México", 15, "CONCACAF", "A"], ["EE. UU.", 16, "CONCACAF", "D"],
    ["Uruguay", 17, "CONMEBOL", "H"], ["Japón", 18, "AFC", "F"], ["Suiza", 19, "UEFA", "B"],
    ["Irán", 21, "AFC", "G"], ["Turquía", 22, "UEFA", "D"], ["Ecuador", 23, "CONMEBOL", "E"],
    ["Austria", 24, "UEFA", "J"], ["Corea del Sur", 25, "AFC", "A"], ["Australia", 27, "AFC", "D"],
    ["Argelia", 28, "CAF", "J"], ["Egipto", 29, "CAF", "G"], ["Canadá", 30, "CONCACAF", "B"],
    ["Noruega", 31, "UEFA", "I"], ["Panamá", 33, "CONCACAF", "L"], ["Costa de Marfil", 34, "CAF", "E"],
    ["Suecia", 38, "UEFA", "F"], ["Paraguay", 40, "CONMEBOL", "D"], ["República Checa", 41, "UEFA", "A"],
    ["Escocia", 43, "UEFA", "C"], ["Túnez", 44, "CAF", "F"], ["RD Congo", 46, "CAF", "K"],
    ["Uzbekistán", 50, "AFC", "K"], ["Catar", 55, "AFC", "B"], ["Irak", 57, "AFC", "I"],
    ["Sudáfrica", 60, "CAF", "A"], ["Arabia Saudita", 61, "AFC", "H"], ["Jordania", 63, "AFC", "J"],
    ["Bosnia y Herzegovina", 65, "UEFA", "B"], ["Cabo Verde", 69, "CAF", "H"], ["Ghana", 74, "CAF", "L"],
    ["Curazao", 82, "CONCACAF", "E"], ["Haití", 83, "CONCACAF", "C"], ["Nueva Zelanda", 85, "OFC", "G"]
]

df = pd.DataFrame(data, columns=["Seleccion", "Ranking_FIFA", "Confederacion", "Grupo"])
csv_content = df.to_csv(index=False)
print(csv_content)