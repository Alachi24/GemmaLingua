#!/bin/sh
MAVEN_HOME=".mvn/wrapper"
MAVEN_OPTS="${MAVEN_OPTS:-}"
WRAPPER_JAR=".mvn/wrapper/maven-wrapper.jar"
if [ ! -f "$WRAPPER_JAR" ]; then
    echo "Maven wrapper JAR not found: $WRAPPER_JAR"
    exit 1
fi
exec java $MAVEN_OPTS -jar "$WRAPPER_JAR" "$@"
