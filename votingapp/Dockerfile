FROM eclipse-temurin:17 AS builder

WORKDIR /app

COPY build.gradle settings.gradle gradlew ./
COPY gradle gradle
RUN chmod +x gradlew
RUN ./gradlew dependencies

COPY src src
RUN ./gradlew build

FROM eclipse-temurin:17

WORKDIR /app

COPY --from=builder /app/build/libs/votingapp.jar app.jar

EXPOSE 8081

CMD ["java", "-jar", "app.jar"]
