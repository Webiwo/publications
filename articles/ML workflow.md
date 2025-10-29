
# 📘 Machine Learning Worklow i przydatne kompetencje


Uczenie maszynowe (Machine Learning) i sztuczna inteligencja (Artificial Intelligence) to bardzo, bardzo, bardzo szerokie pojęcie. Im więcej się dowiaduję o ML/AI, im bardziej wchodzę w szczegóły, tym bardziej zdaję sobie sprawę jak niewiele wiem. Często ludzie przedstawiają ML/AI jako coś magicznego. Aż dziw bierze, że pod spodem jest czysta matematyka i algorytmy ulepszane przez ludzi od dziesięcioleci. Wystarczy tu wspomnieć pierwszy, najprostszy model sztucznej sieci nuronowej, który wprowdził w 1958 roku Frank Rosenblatt - to wydarzyło się prawie 70 lat temu!
Dzisiaj możemy korzystać z tych algorytmów ze względu na moc obliczeniową procesorów i jednostek graficznych GPU, ale aby osiągnąć oczekiwane resultaty musimy zastosować się do pewnego uporządkowanego ciągu kroków i operacji, które prowadzą od surowych danych do działającego modelu ML i jego wdrożenia oraz posiadać też pewne kompetencje, które mogą być bardzo pomocne w ich realizacji. I o tym będzie mowa, o 6 krokach do osiągnięcia określonego celu z wykorzystaniem uczenia maszynowego.


<img width="1492" height="949" alt="image" src="https://github.com/user-attachments/assets/5f55c7f9-49a3-4ff8-954f-ed513571b666" />


---

## Spis treści
1. [Definicja problemu](#definicja-problemu)
2. [Zbieranie danych](#zbieranie-danych)
3. [Ocena rezultatu](#ocena-rezultatu)
4. [Przetwarzanie danych](#przetwarzanie-danych)
5. [Modelowanie i ewaluacja](#modelowanie-i-ewaluacja)
6. [Eksperymentowanie](#eksperymentowanie)
---

## **Definicja problemu**
Jaki problem biznesowy próbujemy rozwiązać? Czy potrzebujemy w ogóle ML/AI do jego rozwiązania. Jeśli tak to w jaki sposób można go sformułować jako problem uczenia maszynowego?
Musimy dobrze zrozumieć potrzebę biznesową, przeanalizować sytuację, zdefiniować cel i metody jego osiągnięcia. W tym może nam pomóc klasyczna analiza biznesowa i narzędzia, których używają właściciele produktów jak np. Impact Mapping.


---

## **Zbieranie danych**
Jeśli uczenie maszynowe polega na wydobywaniu wniosków z danych, to musimy odpowiedzieć na pytanie jakich danych potrzebujemy. Jakie dane już mamy i  w jaki sposób odpowiadają one zdefiniowanemu problemowi? Jakich danych nam brakuje i skąd je weźmiemy? Czy nasze dane są ustrukturyzowane czy nieustrukturyzowane? Statyczne czy strumieniowe? W jaki sposób dokonamy próbkowania tych danych: losowo czy może warstwowo? Czy dane, które posiadamy nie zostały już jakoś wcześniej przetworzone – zaokrąglone a może ograniczone do jakiejś dolnej lub górnej granicy z powodów biznesowych, które kiedyś były istotne? Jednym słowem mówiąc – czy mamy odpowiednie, wiarygodne i reprezentatywne dane, które odpowiadają na pytania zdefiniowane w fazie Problem Definition. Trzeba je dobrze zrozumieć, porozmawiać z ekspertami, sprawdzić ich jakość, spójność i kompletność bo od tego zależy osiągnięcie określonego wcześniej celu. Brzmi trochę jak zarządzanie produktem połączone z zarządzaniem projektem, analizą biznesową i zapewnieniem jakości. Można jednak powiedzieć, że jest to klasyczny problem organizacji, które rzeczywiście podejmują decyzje w oparciu o dane.


---

## **Ocena rezultatu**
Trochę odpowiedzi już mamy a może i nie. Może jest więcej znaków zapytania niż odpowiedzi. Bazując na tym co już wiemy musimy odpowiedzieć sobie teraz na kluczowe pytanie – co definiuje nasz sukces. Jakie wyniki chcemy uzyskać, o jakiej dokładności? Czy dokładność 95% jest wystarczająco dobra?
Zaraz, zaraz – ale czy tylko o dokładność procentową nam chodzi. Jeżeli zastanowimy się głębiej to model może przewidzieć wynik negatywny, podczas gdy w rzeczywistości powinien być pozytywny.
W niektórych przypadkach, jak przy klasyfikacji spamu w e-mailach, takie wyniki nie są dużym problemem. Jednak jeśli system wizyjny samochodu autonomicznego przewidzi brak pieszego, podczas gdy faktycznie znajduje się na jezdni — to już poważny błąd.
W przypadku problemów regresyjnych, czyli gdy chcemy przewidzieć wartość liczbową naszym celem jest zminimalizowanie różnicy między przewidywaniami modelu a rzeczywistymi wartościami. Na przykład, jeśli firma ubezpieczeniowa próbuje przewidzieć cenę ubezpieczenia, które chce zaoferować swoim klientom, to zależy jej przecież, aby model podał wartość jak najbliższą rzeczywistej cenie rynkowej, biorąc pod uwagę wiele różnych czynników, od których ta cena zależy.


---

## **Przetwarzanie danych**
Teraz trochę programowania, odkrywania tego co kryje się w naszym zbiorze danych i przeprowadzenie różnego typu transformacji – od łączenia danych pochodzących z różnych źródeł, przez rozwiązywanie różnych konfliktów, poprawianie błędów, standaryzację, tworzenie nowych cech i usuwanie nieistotnych. I ponownie wiele decyzji do podjęcia bo dane mogą być zaszumione, mogą być niekompletne, mogą być niezbalansowane, mogą być zaprezentowane w różnej skali, mogą być numeryczne lub nie - a jak dobrze wiemy komputer rozumie tylko liczby.
W końcu może wystąpić bardzo trywialny problem – danych może być po prostu za mało. A to wszystko może bardzo wpłynąć na osiągnięcie celu biznesowego. Więc przed nami dużo żmudnej, ale za to bardzo cennej pracy. Z pewnością przydadzą się tu umiejętności analityczne, programistyczne i zdolności komunikacyjne. Jest to bardzo ważny etap, który zajmuje często 70–80% całej pracy i może zdecydować o wyniku końcowym a zatem nie można go potraktować po macoszemu. Jakość, jakość, jakość – od jakości danych zależy wszystko.


---

## **Modelowanie i ewaluacja**
I dochodzimy do serca całego projektu uczenia maszynowego. Teraz trzeba tylko wybrać odpowiednią technikę bądź kilka różnych technik, aby porównać wyniki i wybrać najlepsze rozwiązanie, podzielić dane na zbiory no i wytrenować model. No i w końcu można iść na kawę. Wydaje się proste, ale w rzeczywistości wcale takie nie jest. Proces uczenia trzeba nadzorować – trochę jak nadzoruje się projekt – trzeba monitorować metryki błędu i postęp uczenia, może trzeba go będzie nawet zatrzymać. Może będziemy musieli się zastanowić nad jakimś kompromisem między dokładnością, złożonością a szybkością działania. A może model będzie bardzo dobrze uczył się na podstawie danych treningowych ale będzie bezużyteczny jeśli dostarczymy mu dane, których jeszcze nie widział. No i okaże się, że zamiast spodziewanej dokładności na poziomie 95% otrzymamy 75%. Hmmm. I co teraz począć? Trzeba przejść do następnej fazy.


---

## **Eksperymentowanie**
Tu przyda się zwinne podejście do tematu. Przydadzą się nam do tego etapu wszystkie pomiary i metryki jakościowe. Dzięki nim będziemy wiedzieli np. czy model jest przetrenowany czy może niedotrenowany, co pozwoli na odpowiednie dostrojenie modelu. Może trzeba będzie poeksperymentować z danymi, aby w trakcie uczenia modelu dostarczać mu różne porcje danych. Może czaka nas kolejna porcja transformacji danych np. przy użyciu techniki augmentacji, czyli sztucznego zwiększenia liczby przykładów w zbiorze danych przez stworzenie nowych, zmodyfikowanych wersji istniejących danych treningowych. A może trzeba będzie porzucić dotychczasową drogę i zacząć od początku. Uczymy się na obserwacjach, popełnionych błędach i wdrażamy proces ciągłego doskonalenia (Plan-Do-Check-Act). Zastanawiamy się jak zmienić dotychczasowe kroki w oparciu o to co odkryliśmy i przystępujemy do kolejnej iteracji. Chciałoby się rzec - do kolejnego sprintu, bazując na terminologii Srcum. W końcu po kilku iteracjach może się okazać, że osiągnęliśmy sukces, ale może też być zupełnie odwrotnie. Wówczas trzeba będzie zaprezentować wyniki osobom odpowiedzialnym za podejmowanie kluczowych decyzji w organizacji – a więc po serii analiz konieczna będzie synteza i prezentacja zebranych wyników, czyli stworzenie holistycznego obrazu sytuacji, który pomoże w podjęciu strategicznych decyzji biznesowych.


---

📅 _Data utworzenia:_ 2025-10-28  
✍️ _Autor:_ Sławomir Piwowarski



