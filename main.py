from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()

if __name__ == "__main__":
    schumacher_info = """
    Michael Schumacher[a] (born 3 January 1969) is a German former racing driver who competed in Formula One from 1991 to 2006 and from 2010 to 2012. Schumacher won a record-setting seven Formula One World Drivers' Championship titles, tied by Lewis Hamilton in 2020, and—at the time of his retirement—held the records for most wins (91), pole positions (68), and podium finishes (155), while he maintains the record for most fastest laps (77), among others.
    Born in Hürth to a working-class family, Schumacher began competitive kart racing aged four in a pedal kart built from discarded parts. After a successful karting career—culminating in his victory at the European Championship in 1987—Schumacher graduated to junior formulae. He dominated Formula König in his debut season, before graduating to German Formula Three in 1989, where he finished third. He won the title in 1990, also claiming the Macau Grand Prix and becoming a race-winner in the World Sportscar Championship with Sauber Mercedes. Schumacher made his debut Formula One appearance with Jordan at the Belgian Grand Prix in 1991; his qualifying performance saw Benetton sign him for the remainder of the season. In 1992, he achieved his maiden victory in Belgium amongst several podiums, which he repeated at the Portuguese Grand Prix in 1993. Schumacher won his maiden World Drivers' Championship with eight victories in 1994, following a collision with his rival Damon Hill at the last race of the season. He won a further nine Grands Prix as he defended his title in 1995.
    Schumacher moved to the struggling Ferrari for his 1996 campaign, where he took several victories and finished third overall. He was involved in title battles in 1997 and 1998, being disqualified from the former for a collision with Jacques Villeneuve and finishing runner-up to Mika Häkkinen in the latter. His rivalry with Häkkinen continued into 1999, when Schumacher broke his leg following a brake failure whilst second in the championship. He returned to beat Häkkinen to his first title with Ferrari in 2000, their first in 21 years, which he successfully defended in 2001. His 2002 campaign—during which he won a then-record 11 Grands Prix—saw him claim a record-equalling fifth title with an unparalleled perfect podium rate. He then claimed his unprecedented sixth and seventh titles in 2003 and 2004, holding off Kimi Räikkönen and Juan Pablo Montoya in the former before winning 13 of 18 Grands Prix during the latter, breaking several further records. After dropping to third in 2005 and narrowly finishing runner-up to Fernando Alonso in 2006, Schumacher announced his retirement from Formula One. He later returned with the resurrected Mercedes from 2010 to 2012, claiming his final podium at the latter European Grand Prix, and has been credited with elevating the project to championship-winning form.
    Schumacher was noted for pushing his machinery to the limit for sustained periods, as well as his pioneering fitness regimen, win-at-all-costs mentality, and ability to galvanise teams around him. Appointed a UNESCO Champion for Sport in 2002, Schumacher has been involved in several humanitarian projects and has donated over US$65 million to various charities. In December 2013, Schumacher suffered a traumatic brain injury in a skiing accident and was placed in an induced coma for six months. He received further rehabilitation in Lausanne before being relocated to receive private treatment at his home in September 2014; he has not appeared publicly since.
    """
    lovelace_info = """
    Augusta Ada King, Countess of Lovelace (née Byron; 10 December 1815 – 27 November 1852), also known as Ada Lovelace, was an English mathematician and writer chiefly known for work on Charles Babbage's proposed mechanical general-purpose computer, the analytical engine. She was the first to recognise the machine had applications beyond pure calculation. Lovelace is often considered the first computer programmer.
    Lovelace was the only legitimate child of poet Lord Byron and reformer Anne Isabella Milbanke.[2] Lord Byron separated from his wife a month after Ada was born, and died when she was eight. Although often ill in childhood, Lovelace pursued her studies assiduously. She married William King in 1835. King was a Baron, and was created Viscount Ockham and 1st Earl of Lovelace in 1838. The name Lovelace was chosen because Ada was descended from the extinct Baron Lovelaces.[3] The title given to her husband thus made Ada the Countess of Lovelace.
    Lovelace's educational and social exploits brought her into contact with scientists such as Andrew Crosse, Charles Babbage, David Brewster, Charles Wheatstone and Michael Faraday, and the author Charles Dickens, contacts which she used to further her education. Lovelace described her approach as "poetical science"[4] and herself as an "Analyst (& Metaphysician)".[5]
    At age 18, Lovelace's mathematical talents led her to a long working relationship and friendship with fellow British mathematician Charles Babbage. She was particularly interested in Babbage's work on the analytical engine. Lovelace first met him on 5 June 1833, when she and her mother attended one of Charles Babbage's Saturday night soirées[6] with their mutual friend, and Lovelace's private tutor, Mary Somerville. Though Babbage's analytical engine was never constructed and did not influence the invention of electronic computers, it has been recognised as a Turing-complete general-purpose computer, which anticipated the essential features of a modern electronic computer. Babbage is therefore known as the "father of computers," and Lovelace is credited with several computing "firsts" for her collaboration with him. Lovelace translated an article by the military engineer Luigi Menabrea about the analytical engine, supplementing it with seven long explanatory notes. These described a method of using the machine to calculate Bernoulli numbers which is often called the first published computer program.
    She developed a vision of the capability of computers to go beyond mere calculating or number-crunching, while many others, including Babbage, focused only on those capabilities.[7] Lovelace was the first to point out the possibility of encoding information besides mere arithmetical figures, such as music, and manipulating it with such a machine. Her mindset of "poetical science" led her to ask questions about the analytical engine, examining how individuals and society relate to technology as a collaborative tool.[8] Ada is widely commemorated, including in the names of a programming language, roads, buildings and institutes, as well as programmes, lectures and courses. There are plaques, statues, paintings, literary and non-fiction works about her.
    """
    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary.
    2. Two interesting facts about them.
    """

    prompt_template_summary = PromptTemplate(
        template=summary_template,
        input_variables=["information"],
    )

    llm = ChatAnthropic(model="claude-sonnet-5")
    # llm = ChatOllama(model="gemma4:e4b-mlx")
    chain = prompt_template_summary | llm
    response = chain.invoke(input={"information": lovelace_info})

    print(response.content)
