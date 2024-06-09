//
//  PostUdacityResponse.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 08/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation

struct account: Codable {
    let registered: Bool
    let key: String
}

struct session: Codable {
    let id: String
    let expiration: String
}

struct PostUdacityResponse: Codable {
    let account: account
    let session: session
    
    enum CodingKeys: String, CodingKey {
        case account
        case session
    }
}
