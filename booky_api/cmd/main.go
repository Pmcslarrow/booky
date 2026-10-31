package main

import (
	"fmt"
	"log"
	"net"

	"bookyapi/internal/api/handlers"
	"bookyapi/internal/interceptors"
	booky "bookyapi/proto"

	"google.golang.org/grpc"
)

const (
	PORT = 8080
)

func main() {
	lis, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", PORT))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	// Options
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(interceptors.UnaryLoggingInterceptor),
	)

	// Handlers
	authHandler := handlers.NewAuthHandler()

	// Registrations
	booky.RegisterAuthServiceServer(grpcServer, authHandler)

	log.Printf("gRPC server listening on port %d...", PORT)
	err = grpcServer.Serve(lis)
	if err != nil {
		log.Fatalf("err listening: %v", err)
	}
}
