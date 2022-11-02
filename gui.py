from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import main

db_cards=[]
CARD_SIZE=180
CARDS_PER_ROW=3

class HomeScreen():

    def __init__(self, root):
        self.root = root
        self.root.geometry("1400x830")
        self.root.resizable(0, 0)

        self.container = Frame(root)
        self.canvas = Canvas(self.container)
        self.scrollbar = Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = Scrollbar(self.container, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set)

        self.initialise_variable()
        self.initialise_elements()
        self.place_elements()

    def initialise_elements(self):
        self.leftImage=Label(self.root,image=leftImage)
        self.rightImage=Label(self.root,image=rightImage)
        self.choose_image_button = Button(
            self.root, takefocus=0, text="Choose Image", relief="raised", command=self.choose_image_button_pressed)
        self.query_button = Button(
            self.root, takefocus=0, text="Query", relief="raised", command=self.query_button_pressed)
        self.reset_button = Button(
            self.root, takefocus=0, text="Reset", relief="raised", command=self.reset_button_pressed)
        self.extra1_button = Button(
            self.root, takefocus=0, text="Extra 1", relief="raised", command=self.extra1_button_pressed)
        self.extra2_button = Button(
            self.root, takefocus=0, text="Extra 2", relief="raised", command=self.extra2_button_pressed)
    
    def place_elements(self):
        self.leftImage.place(
            relx=10/1400, rely=10/830, relwidth=350/1400, relheight=350/830)
        self.rightImage.place(
            relx=10/1400, rely=400/830, relwidth=350/1400, relheight=350/830)

        self.choose_image_button.place(
            relx=400/1400, rely=10/830, relwidth=150/1400, relheight=50/830)
        self.query_button.place(
            relx=600/1400, rely=10/830, relwidth=150/1400, relheight=50/830)
        self.reset_button.place(
            relx=800/1400, rely=10/830, relwidth=150/1400, relheight=50/830)
        self.extra1_button.place(
            relx=1000/1400, rely=10/830, relwidth=150/1400, relheight=50/830)
        self.extra2_button.place(
            relx=1200/1400, rely=10/830, relwidth=150/1400, relheight=50/830)

        self.container.place(
            relx=400/1400, rely=100/830, relwidth=950/1400, relheight=650/830)
        self.scrollbar.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # for i in range(50):
        #     self.add_db_card(i,"/Users/harshagarwal/Desktop/ML_SOP/code/util/optic_nerve.png")

    def initialise_variable(self):
        global leftImage,rightImage,leftImagePath,rightImagePath
        leftImagePath=rightImagePath="/Users/harshagarwal/Desktop/ML_SOP/code/default.png"
        leftImage = ImageTk.PhotoImage(Image.open("/Users/harshagarwal/Desktop/ML_SOP/code/default.png"))
        rightImage = ImageTk.PhotoImage(Image.open("/Users/harshagarwal/Desktop/ML_SOP/code/default.png"))
    
    def choose_image_button_pressed(self):
        filename = askopenfilename()
        global leftImage,leftImagePath
        try:
            print(filename)
            resized=Image.open(filename).resize((350, 350),Image.ANTIALIAS)
            self.reset_button_pressed()
            leftImagePath=filename
            leftImage = ImageTk.PhotoImage(resized)
            self.leftImage.configure(image=leftImage)
            self.leftImage.image=leftImage
        except Exception as e:
            print(str(e))

    def query_button_pressed(self):
        global leftImagePath
        res=main.query(leftImagePath)
        self.clear_db_cards()
        for x in res:
            self.add_db_card(x[0],x[1])

    def reset_button_pressed(self):
        global leftImage,rightImage,leftImagePath,rightImagePath
        leftImagePath=rightImagePath="/Users/harshagarwal/Desktop/ML_SOP/code/default.png"
        leftImage = rightImage = ImageTk.PhotoImage(Image.open("/Users/harshagarwal/Desktop/ML_SOP/code/default.png"))
        self.leftImage.configure(image=leftImage)
        self.leftImage.image=leftImage
        self.rightImage.configure(image=rightImage)
        self.rightImage.image=rightImage
        self.clear_db_cards()

    def extra1_button_pressed(self):
        pass

    def extra2_button_pressed(self):
        pass

    def add_db_card(self,dist,source):
        f=Frame(self.scrollable_frame)
        distLabel=Label(f,text="D = "+str(dist))
        sourceLabel=Label(f,text=source.split("/")[-1])
        db_cards.append([dist,source,ImageTk.PhotoImage(Image.open(source).resize((CARD_SIZE, CARD_SIZE),Image.ANTIALIAS)),f])
        imgLabel=Label(f,image=db_cards[-1][2],width=CARD_SIZE, height=CARD_SIZE,takefocus=True)
        imgLabel.pack()
        sourceLabel.pack()
        distLabel.pack()
        k=len(db_cards)-1
        f.grid(row=int(k/CARDS_PER_ROW)+1,column=(k%CARDS_PER_ROW)+1,padx=20,pady=20)
        imgLabel.bind("<Button-1>",lambda event, arg={"i":k}: self.db_card_clicked(event, arg))
    
    def clear_db_cards(self):
        global db_cards
        for x in db_cards:
            x[3].destroy()
        db_cards=[]
    
    def db_card_clicked(self,event,args):
        print("Card Clicked")
        card=db_cards[args["i"]]
        global rightImagePath,rightImage
        rightImagePath=card[1]
        rightImage=ImageTk.PhotoImage(Image.open(rightImagePath).resize((350, 350),Image.ANTIALIAS))
        self.rightImage.configure(image=rightImage)
        self.rightImage.image=rightImage



root=Tk()
root.title("Retinal CBIR")
HomeScreen(root)
root.mainloop()
