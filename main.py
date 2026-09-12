from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

load_dotenv()

if __name__ == "__main__":
    info = """
    Michael Schumacher[a] (born 3 January 1969) is a German former racing driver who competed in Formula One from 1991 to 2006 and from 2010 to 2012. Schumacher won a record-setting seven Formula One World Drivers' Championship titles, tied by Lewis Hamilton in 2020, and—at the time of his retirement—held the records for most wins (91), pole positions (68), and podium finishes (155), while he maintains the record for most fastest laps (77), among others.
    Born in Hürth to a working-class family, Schumacher began competitive kart racing aged four in a pedal kart built from discarded parts. After a successful karting career—culminating in his victory at the European Championship in 1987—Schumacher graduated to junior formulae. He dominated Formula König in his debut season, before graduating to German Formula Three in 1989, where he finished third. He won the title in 1990, also claiming the Macau Grand Prix and becoming a race-winner in the World Sportscar Championship with Sauber Mercedes. Schumacher made his debut Formula One appearance with Jordan at the Belgian Grand Prix in 1991; his qualifying performance saw Benetton sign him for the remainder of the season. In 1992, he achieved his maiden victory in Belgium amongst several podiums, which he repeated at the Portuguese Grand Prix in 1993. Schumacher won his maiden World Drivers' Championship with eight victories in 1994, following a collision with his rival Damon Hill at the last race of the season. He won a further nine Grands Prix as he defended his title in 1995.
    Schumacher moved to the struggling Ferrari for his 1996 campaign, where he took several victories and finished third overall. He was involved in title battles in 1997 and 1998, being disqualified from the former for a collision with Jacques Villeneuve and finishing runner-up to Mika Häkkinen in the latter. His rivalry with Häkkinen continued into 1999, when Schumacher broke his leg following a brake failure whilst second in the championship. He returned to beat Häkkinen to his first title with Ferrari in 2000, their first in 21 years, which he successfully defended in 2001. His 2002 campaign—during which he won a then-record 11 Grands Prix—saw him claim a record-equalling fifth title with an unparalleled perfect podium rate. He then claimed his unprecedented sixth and seventh titles in 2003 and 2004, holding off Kimi Räikkönen and Juan Pablo Montoya in the former before winning 13 of 18 Grands Prix during the latter, breaking several further records. After dropping to third in 2005 and narrowly finishing runner-up to Fernando Alonso in 2006, Schumacher announced his retirement from Formula One. He later returned with the resurrected Mercedes from 2010 to 2012, claiming his final podium at the latter European Grand Prix, and has been credited with elevating the project to championship-winning form.
    Schumacher was noted for pushing his machinery to the limit for sustained periods, as well as his pioneering fitness regimen, win-at-all-costs mentality, and ability to galvanise teams around him. Appointed a UNESCO Champion for Sport in 2002, Schumacher has been involved in several humanitarian projects and has donated over US$65 million to various charities. In December 2013, Schumacher suffered a traumatic brain injury in a skiing accident and was placed in an induced coma for six months. He received further rehabilitation in Lausanne before being relocated to receive private treatment at his home in September 2014; he has not appeared publicly since.
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
    chain = prompt_template_summary | llm
    response = chain.invoke(input={"information": info})

    print(response.content)
